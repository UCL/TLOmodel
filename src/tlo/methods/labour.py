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
logger.setLevel(logging.DEBUG)


class Labour (Module):
    """This module for labour, delivery, the immediate postpartum period and skilled birth attendance."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary will store additional information around delivery and birth
        self.mother_and_newborn_info = dict()

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
            Types.LIST, 'probability a maternal hemorrhage is non-severe (<1000mls) or severe (>1000mls) '),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_obstructed_labour': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother survives'),
        'prob_still_birth_antepartum_haem': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother survives'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother survives'),
        'prob_still_birth_uterine_rupture': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother survives'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother survives'),
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
        'rr_newborn_sepsis_proph_abx': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following prophylactic antibiotics for '
                        'premature labour in a facility'),
        'rr_pph_amtsl': Parameter(
            Types.REAL, 'relative risk of severe post partum haemorrhage following active management of the third '
                        'stage of labour'),
        'prob_successful_manual_removal_placenta': Parameter(
            Types.REAL, 'probability of manual removal of retained products arresting a post partum haemorrhage'),
        'success_rate_pph_surgery': Parameter(
            Types.REAL, 'probability of surgery for postpartum haemorrhage being successful'),
        'success_rate_uterine_repair': Parameter(
            Types.REAL, 'probability repairing a ruptured uterus surgically'),
        'prob_successful_assisted_vaginal_delivery': Parameter(
            Types.REAL, 'probability of successful assisted vaginal delivery'),
        'dummy_prob_health_centre': Parameter(
            Types.REAL, 'dummy probability that a level 1 facility is a health centre'),
        'squeeze_factor_threshold_delivery_attendance': Parameter(
            Types.REAL, 'dummy squeeze factor threshold after which delivery will not be attended '),
        'squeeze_factor_threshold_sba_did_not_run': Parameter(
            Types.REAL, 'dummy squeeze factor threshold after SBA HSI did not run '),
        'sensitivity_of_assessment_of_obstructed_labour_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a health '
                        'centre'),
        'sensitivity_of_assessment_of_obstructed_labour_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_obstructed_labour_for_cs': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant that for obstructed labour in a level 1 '
                        'facility a women needs a caesarean'),
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
            Types.REAL, 'DALY weight for obstructed labour'),
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
        'la_obstructed_labour_treatment': Property(Types.CATEGORICAL, 'If this woman has received treatment for '
                                                                      'obstructed labour, and how promptly',
                                                   categories=['none', 'prompt_treatment', 'delayed_treatment']),
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
        'la_sepsis_treatment': Property(Types.CATEGORICAL, 'If this woman has received treatment for maternal sepsis'
                                                           ' and how promptly',
                                                           categories=['none', 'prompt_treatment',
                                                                       'delayed_treatment']),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_eclampsia_postpartum': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure '
                                                        'following this delivery'),
        'la_eclampsia_disab': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        'la_eclampsia_treatment': Property(Types.BOOL, 'whether this womans uterine rupture has been treated'),
        'la_severe_pre_eclampsia_treatment': Property(Types.BOOL, 'whether this woman has been treated for severe '
                                                                  'pre-eclampsia'),
        'la_maternal_hypertension_treatment': Property(Types.BOOL, 'whether this woman has been treated for maternal '
                                                                   'hypertension'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_treatment': Property(Types.CATEGORICAL, 'If this woman has received treatment for '
                                                                    'postpartum haemorrhage and how promptly',
                                                                    categories=['none', 'prompt_treatment',
                                                                                'delayed_treatment']),
        'la_maternal_haem_non_severe_disab': Property(Types.BOOL, 'disability associated with non severe maternal '
                                                                  'haemorrhage'),
        'la_maternal_haem_severe_disab': Property(Types.BOOL, 'disability associated with severe maternal haemorrhage'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_maternal_death_in_labour': Property(Types.BOOL, ' whether the woman has died as a result of this '
                                                            'pregnancy'),  # DUMMY
        'la_maternal_death_in_labour_date': Property(Types.DATE, 'date of death for a date in pregnancy')  # DUMMY
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        # Here we will include DALY weights if applicable...
        if 'HealthBurden' in self.sim.modules.keys():
            params['la_daly_wts'] = \
                {'haemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339),
                 'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                 'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340),
                 'eclampsia': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861),
                 'obstructed_labour': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348),
                 'uterine_rupture': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                 }
            # n.b. DALY weights for severe maternal haemorrhage and uterine rupture are the same and classified as
            # 'Abdominopelvic problem, severe'. Eclampsia DALY weight doesnt exist- using epilepsy weight:
            # 'Epilepsy, seizures 1-11 per yea'


# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # Here we define the equations that will be used throughout this module using the linear model and stored them
        # as a parameter

        # TODO: finalise predictors in all models

        params['la_labour_equations'] =\
            {'parity': LinearModel(
                LinearModelType.ADDITIVE,
                -3,
                Predictor('age_years').apply(lambda age_years: (age_years * 0.22)),  # params['intercept_parity
                                                                                     # _lr2010'])),
                Predictor('li_mar_stat').when('2', 0.91)  # params['effect_mar_stat_2_parity_lr2010'])
                                        .when('3', 0.16),  # params['effect_mar_stat_3_parity_lr2010']),
                Predictor('li_wealth').when('5', -0.13)  # params['effect_wealth_lev_5_parity_lr2010'])
                                      .when('4', -0.13)  # params['effect_wealth_lev_4_parity_lr2010'])
                                      .when('3', -0.26)  # params['effect_wealth_lev_3_parity_lr2010'])
                                      .when('2', -0.37)  # params['effect_wealth_lev_2_parity_lr2010'])
                                      .when('1', -0.9)),  # params['effect_wealth_lev_1_parity_lr2010'])),
                # TODO: first draft from rough regression of 2010 DHS data (will need to be finalised)
                # TODO: parameters are messing this equation up (sending all parities minus)

             'early_preterm_birth': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_early_ptb'],
                Predictor('age_years').when('.between(15,20)', params['or_early_ptb_age<20']),
                Predictor('la_has_previously_delivered_preterm').when(True, params['or_early_ptb_prev_ptb'])),

             'late_preterm_birth': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_late_ptb'],
                Predictor('la_has_previously_delivered_preterm').when(True, params['or_late_ptb_prev_ptb'])),

             'post_term_birth': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_post_term_labour'],
                Predictor('li_bmi').when('>24', params['rrr_ptl_bmi_more25']),
                Predictor('li_mar_stat').when('1', params['rrr_ptl_not_married'])
                                        .when('3', params['rrr_ptl_not_married'])),

             'obstructed_labour_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pl_ol'],
                Predictor('la_parity').when('0', params['rr_PL_OL_nuliparity']),
                Predictor('la_parity').when('>2', params['rr_PL_OL_para_more3']),
                Predictor('li_bmi').when('<18', params['rr_PL_OL_bmi_less18']),
                Predictor('li_bmi').when('>24', params['rr_PL_OL_bmi_more25']),
                Predictor('age_years').when('<20', params['rr_PL_OL_age_less20'])),

             'obstructed_labour_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_obstructed_labour'],
                Predictor('la_obstructed_labour_treatment').when('prompt_treatment', 0.5),
                Predictor('la_obstructed_labour_treatment').when('delayed_treatment', 0.25),
                Predictor('la_maternal_death_in_labour').when(True, 3)),

             'sepsis_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_sepsis'],
                Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol'])),
                # todo: less data on intrapartum sepsis risk factors? maybe dont include

             'sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_sepsis'],
                Predictor('la_sepsis_treatment').when('prompt_treatment', 0.5),
                Predictor('la_sepsis_treatment').when('delayed_treatment', 0.25)),

             'sepsis_pp': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_pp_sepsis'],
                Predictor('li_urban').when(False, params['or_pp_sepsis_rural']),
                Predictor('li_ed_lev').when('1', params['or_pp_sepsis_no_edu']),
                Predictor('li_ed_lev').when('2', params['or_pp_sepsis_primary_edu']),
                Predictor('delivery_type', external=True).when('instrumental', params['or_pp_sepsis_avd'])
                                                         .when('caesarean_section', params['or_pp_sepsis_cs']),
                Predictor('ps_premature_rupture_of_membranes').when(True, params['or_pp_sepsis_prom'])),
                # todo: study doesnt give post-partum time period, this may be better in postpartum module?

             'sepsis_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_sepsis'],
                Predictor('la_sepsis_treatment').when('prompt_treatment', 0.5),
                Predictor('la_sepsis_treatment').when('delayed_treatment', 0.25)),
                # DUMMY, copy from above

             'sepsis_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_sepsis'],
                Predictor('la_sepsis_treatment').when('prompt_treatment', 0.5),
                Predictor('la_sepsis_treatment').when('delayed_treatment', 0.25),
                Predictor('la_maternal_death_in_labour').when(True, 3)),

             'eclampsia_ip': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['or_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['or_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['or_ip_eclampsia_nullip']),
                Predictor('ps_gest_diab').when(True, params['or_ip_eclampsia_gest_diab']),
                Predictor('la_severe_pre_eclampsia_treatment').when(True, 0.5)),
                # todo: study combines eclampsia and pre-eclampsia as primary outcome?
                # todo: chronic htn strong predictor

             'eclampsia_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, 0.5)),

             'eclampsia_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, 0.5),
                Predictor('la_maternal_death_in_labour').when(True, 3)),
                 
             'eclampsia_pp': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['or_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['or_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['or_ip_eclampsia_nullip']),
                Predictor('ps_gest_diab').when(True, params['or_ip_eclampsia_gest_diab']),
                Predictor('la_severe_pre_eclampsia_treatment').when(True, 0.5)),

             'eclampsia_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, 0.5)),

             'antepartum_haem_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_aph'],
                Predictor('la_obstructed_labour').when(True, params['rr_aph_pl_ol'])),

             'antepartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_aph'],
                Predictor('la_antepartum_haem_treatment').when(True, 0.5),
                Predictor('received_blood_transfusion', external=True).when(True, 0.20)),

             'antepartum_haem_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_antepartum_haem'],
                Predictor('la_antepartum_haem_treatment').when(True, 0.5),
                Predictor('la_maternal_death_in_labour').when(True, 3)),

             'postpartum_haem_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pph'],
                Predictor('la_obstructed_labour').when(True, params['rr_pph_pl_ol'])),

             'postpartum_haem_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_pph'],
                Predictor('la_postpartum_haem_treatment').when('delayed_treatment', 0.25),
                Predictor('la_postpartum_haem_treatment').when('prompt_treatment', 0.5),
                Predictor('received_blood_transfusion', external=True).when(True, 0.20)),

             'uterine_rupture_ip': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_uterine_rupture'],
                Predictor('la_parity').when('>4', params['or_ur_grand_multip']),
                Predictor('la_previous_cs_delivery').when(True, params['or_ur_prev_cs']),
                Predictor('la_obstructed_labour').when(True, params['or_ur_ref_ol'])),
                # todo: should la_obstructed_labour_treatment be a predictor that reduces likelihood of uterine rupture?

             'uterine_rupture_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_uterine_rupture'],
                Predictor('la_uterine_rupture_treatment').when(True, 0.5)),
             
             'uterine_rupture_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_uterine_rupture'],
                Predictor('la_uterine_rupture_treatment').when(True, 0.5),
                Predictor('la_maternal_death_in_labour').when(True, 2)),

             'probability_delivery_health_centre': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_deliver_in_health_centre'],
                Predictor('age_years').when('.between(24,30)', params['rrr_hc_delivery_age_25_29'])
                                      .when('.between(29,35)', params['rrr_hc_delivery_age_30_34'])
                                      .when('.between(34,40)', params['rrr_hc_delivery_age_35_39'])
                                      .when('.between(39,45)', params['rrr_hc_delivery_age_40_44'])
                                      .when('.between(44,50)', params['rrr_hc_delivery_age_45_49']),
                Predictor('li_urban').when(False, params['rrr_hc_delivery_rural']),
                Predictor('la_parity').when('.between(2,5)', params['rrr_hc_delivery_parity_3_to_4'])
                                      .when('>4', params['rrr_hc_delivery_parity_>4']),
                Predictor('li_mar_stat').when('2', params['rrr_hc_delivery_married'])),
                # TODO: predictors not included due to mismatch between study LR model output and categorisation of TLO
                #  model variables - education, wealth  and marital status

             'probability_delivery_at_home': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_deliver_at_home'],
                Predictor('age_years').when('.between(34,40)', params['rrr_hb_delivery_age_35_39'])
                                      .when('.between(39,45)', params['rrr_hb_delivery_age_40_44'])
                                      .when('.between(44,50)', params['rrr_hb_delivery_age_45_49']),
                Predictor('la_parity').when('.between(2,5)', params['rrr_hb_delivery_parity_3_to_4'])
                                      .when('>4', params['rrr_hb_delivery_parity_>4'])),

                # TODO: as above - wealth, marriage

                'probability_delivery_home': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_careseeking_for_complication'],
                Predictor('li_wealth').when('2', params['or_comp_careseeking_wealth_2'])),

             'care_seeking_for_complication': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_careseeking_for_complication'],
                Predictor('li_wealth').when('2', params['or_comp_careseeking_wealth_2'])),
                # TODO: include ANC
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
        df.loc[df.is_alive, 'la_obstructed_labour_treatment'] = 'none'
        df.loc[df.is_alive, 'la_antepartum_haem'] = False
        df.loc[df.is_alive, 'la_antepartum_haem_treatment'] = False
        df.loc[df.is_alive, 'la_uterine_rupture'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_disab'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_treatment'] = False
        df.loc[df.is_alive, 'la_sepsis'] = False
        df.loc[df.is_alive, 'la_sepsis_postpartum'] = False
        df.loc[df.is_alive, 'la_sepsis_disab'] = False
        df.loc[df.is_alive, 'la_sepsis_treatment'] = 'none'
        df.loc[df.is_alive, 'la_eclampsia'] = False
        df.loc[df.is_alive, 'la_eclampsia_postpartum'] = False
        df.loc[df.is_alive, 'la_eclampsia_disab'] = False
        df.loc[df.is_alive, 'la_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_severe_pre_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_maternal_hypertension_treatment'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_postpartum_haem_treatment'] = 'none'
        df.loc[df.is_alive, 'la_maternal_haem_non_severe_disab'] = False
        df.loc[df.is_alive, 'la_maternal_haem_severe_disab'] = False
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour_date'] = pd.NaT

#  ----------------------------ASSIGNING PARITY AT BASELINE ----------------------------------------------------------

        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = \
            np.around(params['la_labour_equations']['parity'].predict(df.loc[df.is_alive & (df.sex == 'F') &
                                                                             (df.age_years > 14)]))
        df.la_parity.astype(float)

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # ============================= Define consumable for HSIs ====================================================
        # Here all the consumables required within the labour health system interactions are defined and packaged as
        # parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_uncomplicated_delivery = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Vaginal delivery - skilled attendance',
                            'Intervention_Pkg_Code'])[0]

        pkg_code_clean_delivery_kit = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Clean practices and immediate essential newborn care '
                                                               '(in facility)', 'Intervention_Pkg_Code'])[0]

        item_code_abx_prom = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzylpenicillin 1g (1MU), PFR_Each_CMST', 'Item_Code'])[0]

        pkg_code_pprom = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM', 'Intervention_Pkg_Code'])[0]

        item_code_steroids_prem_dexamethasone = pd.unique(
            consumables.loc[consumables['Items'] == 'Dexamethasone 5mg/ml, 5ml_each_CMST', 'Item_Code'])[0]

        item_code_steroids_prem_betamethasone = pd.unique(
            consumables.loc[consumables['Items'] == 'Betamethasone, 12 mg injection', 'Item_Code'])[0]

        item_code_antibiotics_gbs_proph = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzylpenicillin 3g (5MU), PFR_each_CMST', 'Item_Code'])[0]

        pkg_code_severe_preeclampsia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia', 'Intervention_Pkg_Code'])[0]

        pkg_code_obstructed_labour = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of obstructed labour',
                            'Intervention_Pkg_Code'])[0]

        item_code_forceps = pd.unique(consumables.loc[consumables['Items'] == 'Forceps, obstetric', 'Item_Code'])[0]

        item_code_vacuum = pd.unique(consumables.loc[consumables['Items'] == 'Vacuum, obstetric', 'Item_Code'])[0]

        pkg_code_sepsis = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                            'Intervention_Pkg_Code'])[0]

        pkg_code_severe_hypertension = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia', 'Intervention_Pkg_Code'])[0]

        pkg_code_eclampsia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia', 'Intervention_Pkg_Code'])[0]

        pkg_code_am = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Active management of the 3rd stage of labour',
                            'Intervention_Pkg_Code'])[0]

        pkg_code_pph = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum hemorrhage',
                            'Intervention_Pkg_Code'])[0]

        pkg_code_cs = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Cesearian Section with indication (with complication)',
                            'Intervention_Pkg_Code'])[0]

        item_code_blood = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]

        item_code_bt_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]

        item_code_bt_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]

        item_code_bt_gs = pd.unique(
            consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                            'Item_Code'])[0]

        self.consumables ={
            'consumables_prophylaxis_and_attended_delivery': {'Intervention_Package_Code':
                                                            {pkg_code_uncomplicated_delivery: 1,
                                                             pkg_code_clean_delivery_kit: 1, pkg_code_pprom: 1},
             'Item_Code': {item_code_abx_prom: 3, item_code_steroids_prem_dexamethasone: 5,
                           item_code_steroids_prem_betamethasone: 2, item_code_antibiotics_gbs_proph: 3}},

            'consumables_needed_spe': {'Intervention_Package_Code': {pkg_code_severe_preeclampsia: 1},
                                       'Item_Code': {}},

            'consumables_obstructed_labour': {'Intervention_Package_Code': {pkg_code_obstructed_labour: 1},
                                              'Item_Code': {item_code_forceps: 1, item_code_vacuum: 1}},

            'consumables_needed_sepsis': {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}},

            'consumables_needed_htn': {'Intervention_Package_Code': {pkg_code_severe_hypertension: 1},
                                       'Item_Code': {}},

            'consumables_needed_eclampsia': {'Intervention_Package_Code': {pkg_code_eclampsia: 1}, 'Item_Code': {}},

            'consumables_needed_amtsl': {'Intervention_Package_Code': {pkg_code_am: 1}, 'Item_Code': {}},

            'consumables_needed_pph': {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}},

            'consumables_needed_cs': {'Intervention_Package_Code': {pkg_code_cs: 1}, 'Item_Code': {}},

            'consumables_needed_bt': {'Intervention_Package_Code': {}, 'Item_Code': {item_code_blood: 2,
                                                                                     item_code_bt_needle: 1,
                                                                                     item_code_bt_test: 1,
                                                                                     item_code_bt_gs: 2}}
        }

        # =======================Register dx_tests for complications during labour/postpartum=======================
        # We register all the dx_tests needed within the labour HSI events. dx_tests in this module represent assessment
        # and correct diagnosis of key complication, leading to treatment or referral for treatment.

        # Sensitivity of testing varies between health centres and hospitals...

        # todo: could these be shortened with a function to register?

        # Obstructed Labour diagnosis
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_obstructed_labour_hc=DxTest(
                property='la_obstructed_labour',
                sensitivity=self.parameters['sensitivity_of_assessment_of_obstructed_labour_hc'],))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_obstructed_labour_hp=DxTest(
                property='la_obstructed_labour',
                sensitivity=self.parameters['sensitivity_of_assessment_of_obstructed_labour_hp'], ))

        # Sepsis diagnosis intrapartum...
        # dx_tests for intrapartum and postpartum sepsis only differ in the 'property' variable
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_sepsis_hc_ip=DxTest(
                property='la_sepsis',
                sensitivity=self.parameters['sensitivity_of_assessment_of_sepsis_hc'],))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_sepsis_hp_ip=DxTest(
                property='la_sepsis',
                sensitivity=self.parameters['sensitivity_of_assessment_of_sepsis_hp'],))

        # Sepsis diagnosis postpartum
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_sepsis_hc_pp=DxTest(
                property='la_sepsis_postpartum',
                sensitivity=self.parameters['sensitivity_of_assessment_of_sepsis_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_sepsis_hp_pp=DxTest(
                property='la_sepsis_postpartum',
                sensitivity=self.parameters['sensitivity_of_assessment_of_sepsis_hp'], ))

        # Hypertension diagnosis
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_hypertension_hc=DxTest(
                property='ps_currently_hypertensive',
                sensitivity=self.parameters['sensitivity_of_assessment_of_hypertension_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_hypertension_hp=DxTest(
                property='ps_currently_hypertensive',
                sensitivity=self.parameters['sensitivity_of_assessment_of_hypertension_hp'], ))
        # todo: are we ok to turn on ps_currently_hypertensive for new onset eclampsia? Or will it already be true from
        #  PS module

        # severe pre-eclampsia diagnosis
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_severe_pe_hc=DxTest(
                property='ps_severe_pre_eclamp',
                sensitivity=self.parameters['sensitivity_of_assessment_of_severe_pe_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_severe_pe_hp=DxTest(
                property='ps_severe_pre_eclamp',
                sensitivity=self.parameters['sensitivity_of_assessment_of_severe_pe_hp'], ))

        # Antepartum Haemorrhage
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_for_referral_aph_hc=DxTest(
                property='la_antepartum_haem',
                sensitivity=self.parameters['sensitivity_of_referral_assessment_of_antepartum_haem_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_for_treatment_aph_hp=DxTest(
                property='la_antepartum_haem',
                sensitivity=self.parameters['sensitivity_of_treatment_assessment_of_antepartum_haem_hp'], ))

        # Uterine Rupture
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_for_referral_uterine_rupture_hc=DxTest(
                property='la_uterine_rupture',
                sensitivity=self.parameters['sensitivity_of_referral_assessment_of_uterine_rupture_hc'],))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_for_treatment_uterine_rupture_hp=DxTest(
                property='la_uterine_rupture',
                sensitivity=self.parameters['sensitivity_of_treatment_assessment_of_uterine_rupture_hp'], ))

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
        df.at[child_id, 'la_obstructed_labour_treatment'] = 'none'
        df.at[child_id, 'la_antepartum_haem'] = False
        df.at[child_id, 'la_antepartum_haem_treatment'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_uterine_rupture_disab'] = False
        df.at[child_id, 'la_uterine_rupture_treatment'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_sepsis_postpartum'] = False
        df.at[child_id, 'la_sepsis_disab'] = False
        df.at[child_id, 'la_sepsis_treatment'] = 'none'
        df.at[child_id, 'la_eclampsia'] = False
        df.at[child_id, 'la_eclampsia_postpartum'] = False
        df.at[child_id, 'la_eclampsia_disab'] = False
        df.at[child_id, 'la_eclampsia_treatment'] = False
        df.at[child_id, 'la_severe_pre_eclampsia_treatment'] = False
        df.at[child_id, 'la_maternal_hypertension_treatment'] = False
        df.at[child_id, 'la_postpartum_haem'] = False
        df.at[child_id, 'la_postpartum_haem_treatment'] = 'none'
        df.at[child_id, 'la_maternal_haem_non_severe_disab'] = False
        df.at[child_id, 'la_maternal_haem_severe_disab'] = False
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_maternal_death_in_labour'] = False
        df.at[child_id, 'la_maternal_death_in_labour_date'] = pd.NaT

        # If a mothers labour has resulted in an intrapartum still birth her child is still generated by the simulation
        # but the death is recorded through the InstantaneousDeath function

        # Log only live births:
        if ~mother.la_intrapartum_still_birth:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity
            logger.info('%s|live_births|%s',
                        self.sim.date,
                        {'mother': mother_id,
                         'child': child_id,
                         'mother_age': df.at[mother_id, 'age_years']})

        if mother.la_intrapartum_still_birth:
            #  N.B this will only record intrapartum stillbirth
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='intrapartum stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.at[mother_id, 'la_intrapartum_still_birth'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

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
        #    scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
        #                      health_values_df.sum(axis=1)).fillna(1.0)
        #    health_values_df = health_values_df.multiply(scaling_factor, axis=0)

        return health_values_df

    # ===================================== LABOUR SCHEDULER ==========================================================

    def set_date_of_labour(self, individual_id):
        """This function, called by the contraception module, uses linear equations to determine a womans likelihood of
        preterm, postterm or term labour and sets their future date of labour accordingly"""

        df = self.sim.population.props
        params = self.parameters
        logger.debug('person %d is having their labour scheduled on date %s', individual_id, self.sim.date)

        # Check only alive newly pregnant women are scheduled to this function
        assert df.at[individual_id, 'is_alive'] and df.at[individual_id, 'is_pregnant']
        assert df.at[individual_id, 'date_of_last_pregnancy'] == self.sim.date

        # Using the linear equations defined above we calculate this womans individual risk of early and late preterm
        # labour
        eptb_prob = params['la_labour_equations']['early_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]
        lptb_prob = params['la_labour_equations']['late_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]
        ptl_prob = params['la_labour_equations']['post_term_birth'].predict(df.loc[[individual_id]])[individual_id]

        # We then use a random draw to determine if the woman will go into preterm labour and how early she will deliver
        # We store this draw as a variable so the result can be compared against both probabilities
        random_draw = self.rng.random_sample()
        if random_draw < eptb_prob:
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta((self.rng.randint(24, 33)),
                                                                                     unit='W')

        elif random_draw < (lptb_prob + eptb_prob):
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta((self.rng.randint(34, 36)),
                                                                                     unit='W')

        # For women who will deliver after term we apply a risk of post term birth
        else:
            if self.rng.random_sample() < ptl_prob:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta((self.rng.randint(42, 46)),
                                                                                           unit='W')
            else:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta((self.rng.randint(37, 41)),
                                                                                           unit='W')

        # Here we check that no one can go into labour before 24 weeks gestation
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(168, unit='d')

        # and then we schedule the labour for that womans due date
        self.sim.schedule_event(LabourOnsetEvent(self, individual_id),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])

    # ===================================== HELPER FUNCTIONS ==========================================================

    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        person = df.loc[[person_id]]
        if eq == params['la_labour_equations']['antepartum_haem_death'] or eq == params['la_labour_equations'][
                 'postpartum_haem_pp_death']:
            has_rbt = mni[person_id]['received_blood_transfusion']
            return self.rng.random_sample() < eq.predict(person, received_blood_transfusion=has_rbt)[person_id]

        if eq == params['la_labour_equations']['sepsis_pp']:
            mode_of_delivery = mni[person_id]['mode_of_delivery']
            return self.rng.random_sample() < eq.predict(person, delivery_type=mode_of_delivery)[person_id]
        else:
            return self.rng.random_sample() < eq.predict(df.loc[[person_id]])[person_id]

    def set_home_birth_complications_intrapartum(self, individual_id, complication, labour_stage):
        """Uses the result of a linear equation to determine the probability of a certain complications and makes
        changes to the appropriate variables in the data frame dependent on the result."""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        assert mni[individual_id]['delivery_setting'] == 'home_birth'

        if self.eval(params['la_labour_equations'][f'{complication}_{labour_stage}'], individual_id):
            df.at[individual_id, f'la_{complication}'] = True

            if complication == 'antepartum_haem':
                random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                p=params['severity_maternal_haemorrhage'])

                if random_choice == 'non_severe':
                        df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                        df.at[individual_id, 'la_maternal_haem_severe_disab'] = True
            else:
                df.at[individual_id, f'la_{complication}_disab'] = True

            logger.debug(f'person %d has developed {complication} during the intrapartum phase of a homebirth on date'
                         f' %s', individual_id, self.sim.date)
            logger.info(f'%s|{complication}|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                         'person_id': individual_id})

    def set_home_birth_complications_postpartum(self, individual_id, complication, labour_stage):
        """Uses the result of a linear equation to determine the probability of a certain complications and makes
        changes to the appropriate variables in the data frame dependent on the result."""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        if self.eval(params['la_labour_equations'][f'{complication}_{labour_stage}'], individual_id):
            if complication == 'sepsis' or complication == 'eclampsia':
                df.at[individual_id, f'la_{complication}_postpartum'] = True
                df.at[individual_id, f'la_{complication}_disab'] = True

            if complication == 'postpartum_haem':
                df.at[individual_id, f'la_{complication}'] = True
                mni[individual_id]['source_pph'] = self.rng.choice(['uterine_atony', 'retained_placenta'],
                                                                   size=1, p=params['prob_pph_source'])
                random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                p=params['severity_maternal_haemorrhage'])

                if random_choice == 'non_severe':
                    df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[individual_id, 'la_maternal_haem_severe_disab'] = True

            logger.debug(f'person %d has developed {complication} during the postpartum phase of a homebirth on date %s'
                         , individual_id, self.sim.date)
            logger.info(f'%s|{complication}|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                         'person_id': individual_id})

    def calculate_complication_risk_facility_delivery(self, individual_id, complication, labour_stage):
        """This function is called at the beginning of a facility delivery to calculate a woman's risk of each
        complication and store the returned value in her personal MNI dictionary"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        # Check only women delivering at a facility have this function applied
        assert mni[individual_id]['delivery_setting'] == 'health_centre' or mni[individual_id]['delivery_setting']\
               == 'hospital'

        if complication == 'sepsis' and labour_stage == 'pp':
            mode_of_delivery = mni[individual_id]['mode_of_delivery']
            mni[individual_id][f'risk_{labour_stage}_{complication}'] = \
                params['la_labour_equations'][f'{complication}_{labour_stage}'].predict(df.loc[[individual_id]],
                                                                                        delivery_type=mode_of_delivery
                                                                                        )[individual_id]

        else:
            mni[individual_id][f'risk_{labour_stage}_{complication}'] = \
                params['la_labour_equations'][f'{complication}_{labour_stage}'].predict(df.loc[[individual_id]]
                                                                                    )[individual_id]

    def set_complications_during_facility_birth_intrapartum(self, person_id, complication, labour_stage):
        """Using each womans individual risk of a complication (which may have been modified by treatment) this function
        determines if she will experience a complication during her facility delivery. If so, additional treatment is
        scheduled"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        rng = self.rng
        params = self.parameters

        # Ensure women delivering at home are not having complications set this way
        assert mni[person_id]['delivery_setting'] == 'health_centre' or mni[person_id]['delivery_setting'] == 'hospital'

        if rng.random_sample() < mni[person_id][f'risk_{labour_stage}_{complication}']:
            df.at[person_id, f'la_{complication}'] = True
            if complication == 'antepartum_haem':
                random_choice = self.rng.choice(['non_severe', 'severe'], size=1, p=params[
                                                'severity_maternal_haemorrhage'])

                if random_choice == 'non_severe':
                    df.at[person_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[person_id, 'la_maternal_haem_severe_disab'] = True
            else:
                df.at[person_id, f'la_{complication}_disab'] = True

            logger.info(f'%s|{complication}|%s', self.sim.date,
                            {'age': df.at[person_id, 'age_years'],
                             'person_id': person_id})

            logger.debug(f'This is HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1: person %d has '
                             f'developed {complication} during delivery at facility level 1', person_id)

    def set_complications_during_facility_birth_postpartum(self, person_id, complication, labour_stage):
        """"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        rng = self.rng
        params = self.parameters

        if rng.random_sample() < mni[person_id][f'risk_{labour_stage}_{complication}']:
            if complication == 'sepsis' or complication == 'eclampsia':
                df.at[person_id, f'la_{complication}_postpartum'] = True

            # If the woman will experience postpartum bleeding we determine the source of the bleed and store
            # within the mni
            elif complication == 'postpartum_haem':
                df.at[person_id, f'la_{complication}'] = True
                mni[person_id]['source_pph'] = self.rng.choice(['uterine_atony', 'retained_placenta'],
                                                               size=1, p=params['prob_pph_source'])
                random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                p=params['severity_maternal_haemorrhage'])
                if random_choice == 'non_severe':
                    df.at[person_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[person_id, 'la_maternal_haem_severe_disab'] = True
            else:
                df.at[person_id, f'la_{complication}_disab'] = True

            logger.info(f'%s|{complication}|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

            logger.debug(f'This is HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1: person %d has '
                         f'developed {complication} following delivery at facility level 1', person_id)


class LabourOnsetEvent(Event, IndividualScopeEventMixin):
    """This is the LabourOnsetEvent. It is scheduled by the set_date_of_labour function. It represents the start of a
    womans labour. Here we assign a "type" of labour based on gestation (i.e. early preterm), we create a dictionary to
    store additional variables important to labour and HSIs, and we determine if a woman will seek care. This event
    schedules  the LabourAtHome event and the HSI_Labour_PresentsForSkilledAttendance at birth (depending on care
     seeking), the BirthEvent and the LabourDeathEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        # Here we populate the maternal and newborn info dictionary with baseline values before the womans labour begins
        mni = self.module.mother_and_newborn_info
        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d has just reached LabourOnsetEvent on %s', individual_id, self.sim.date)

            # Here we check only women who have reached their due date are going into labour and no women are in labour
            # before 24 weeks
            assert df.at[individual_id, 'la_due_date_current_pregnancy'] == self.sim.date

            # As ps_gestational_age_in_weeks is updated in a weekly event, it is possible for this property to read 23 -
            # when it is actually a womans 24th week of gestation - need a more permanent fix
            assert df.at[individual_id, 'ps_gestational_age_in_weeks'] > 22

            # Here we populate the maternal and newborn info dictionary with baseline values before the womans labour
            # begins
            mni[individual_id] = {'labour_state': None,
                                  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or Post Term (POTL)
                                  'delivery_setting': None,  # home_birth, health_centre, hospital
                                  'delivery_attended': False,  # True or False
                                  'corticosteroids_given': False,
                                  'risk_ip_obstructed_labour': params['prob_pl_ol'],
                                  'risk_ip_sepsis': params['prob_ip_sepsis'],
                                  'risk_pp_sepsis': params['odds_pp_sepsis'],
                                  'sepsis_pp': False,
                                  'risk_ip_antepartum_haem': params['prob_aph'],
                                  'risk_ip_eclampsia': params['odds_ip_eclampsia'],
                                  'risk_pp_eclampsia': params['prob_pp_eclampsia'],
                                  'eclampsia_pp': False,
                                  'risk_ip_uterine_rupture': params['odds_uterine_rupture'],
                                  'risk_pp_postpartum_haem': params['prob_pph'],
                                  'postpartum_haem_pp': False,
                                  'source_pph': None,  # Uterine Atony (UA) or Retained Products/Placenta (RPP)
                                  'risk_newborn_sepsis': params['prob_neonatal_sepsis'],
                                  'risk_newborn_ba': params['prob_neonatal_birth_asphyxia'],
                                  #  Should this just be risk of asphyxia
                                  'mode_of_delivery': 'vaginal_delivery',  # vaginal_delivery, instrumental,
                                                                           # caesarean_section
                                  'sought_care_for_complication': False,
                                  'referred_for_cs': 'none',  # 'none', 'prompt_referral', 'late_referral'
                                  'referred_for_blood': 'none',  # 'none', 'prompt_referral', 'late_referral'
                                  'received_blood_transfusion': False,
                                  'referred_for_surgery': 'none',  # 'none', 'prompt_referral', 'late_referral'
                                  'death_in_labour': False,  # True (T) or False (F)
                                  'cause_of_death_in_labour': [],
                                  'stillbirth_in_labour': False,  # True (T) or False (F)
                                  'cause_of_stillbirth_in_labour': [],
                                  'death_postpartum': False}  # True (T) or False (F)


# ===================================== LABOUR STATE  ==================================================================

        # Debug logging to highlight the women who have miscarried/had an abortion/stillbirth who still come to the
        # event. We also immediately delete their mni
        person = df.loc[individual_id]
        if person.is_alive and ~df.at[individual_id, 'is_pregnant']:
            logger.debug('person %d has just reached LabourOnsetEvent on %s but is no longer pregnant',
                         individual_id, self.sim.date)
            del mni[individual_id]

        # TODO: As PregnancySupervisor event updates GA weekly, some women will present to this event with a gestation
        #  of > threshold number, but their GA hasn't been updated in the df (hence altered GA thresholds below)

        # The event is conditioned on the woman being pregnant, today being her due date and being alive
        # We don't use assert functions to ensure the due date is correct as women who lose their pregnancy will still
        # have this event scheduled
        if person.is_pregnant & person.is_alive & (person.la_due_date_current_pregnancy == self.sim.date):
            df.at[individual_id, 'la_currently_in_labour'] = True

            # Now we use gestational age to categorise the 'labour_state'
            if 37 <= person.ps_gestational_age_in_weeks < 42:
                mni[individual_id]['labour_state'] = 'term_labour'

            # Here we allow a woman to go into early preterm labour with a gestational age of 23 (limit is 24) to
            # account for PregnancySupervisor only updating weekly
            elif 23 <= person.ps_gestational_age_in_weeks < 34:
                mni[individual_id]['labour_state'] = 'early_preterm_labour'
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif 37 > person.ps_gestational_age_in_weeks >= 34:
                mni[individual_id]['labour_state'] = 'late_preterm_labour'
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif person.ps_gestational_age_in_weeks > 41:
                mni[individual_id]['labour_state'] = 'postterm_labour'
                logger.info('%s|postterm_birth|%s', self.sim.date,
                            {'age': df.at[individual_id, 'age_years'],
                             'person_id': individual_id})

            assert mni[individual_id]['labour_state'] is not None
            labour_state = mni[individual_id]['labour_state']
            logger.debug(f'This is LabourOnsetEvent, person %d has now gone into {labour_state} on date %s',
                         individual_id, self.sim.date)

# ===================================== CARE SEEKING AND DELIVERY SETTING =============================================

            # Delivery setting choice during labour is predicted from a multinomial logistic regression. First we
            # predict outcomes using two linear equations
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
                health_centre_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(self.module,
                                                                                                       person_id=
                                                                                                       individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.info('This is LabourOnsetEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on '
                            'date %s for person %d as they have chosen to seek care at a health centre for delivery',
                            self.sim.date, individual_id)
            else:
                non_referral_hospital_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(
                    self.module, person_id=individual_id)
                #  todo: currently we are only scheduling lower level hospital deliveries

                self.sim.modules['HealthSystem'].schedule_hsi_event(non_referral_hospital_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

# ======================================== SCHEDULING BIRTH AND DEATH EVENTS ==========================================

            # We schedule all women to move through the death event where those who have developed a complication
            # that hasn't been treated or treatment has failed will have a case fatality rate applied
            self.sim.schedule_event(LabourDeathEvent(self.module, individual_id), self.sim.date + DateOffset(days=3))

            # Here we schedule the birth event for 4 days after labour- as women who die but still deliver a live child
            # will pass through birth event
            due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
            self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=4))
            logger.debug('This is LabourOnsetEvent scheduling a birth on date %s to mother %d', due_date,
                         individual_id)

            logger.debug('This is LabourOnsetEvent scheduling a potential death on date %s for mother %d',
                         self.sim.date, individual_id)


class LabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """This is the LabourAtHomeEvent. It is scheduled by the LabourOnsetEvent for women who will not seek care. This
    event applies the probability that women delivering at home will experience complications, makes the appropriate
    changes to the data frame . Women who seek care, but for some reason are unable to deliver at a facility will return
    to this event"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        logger.debug('person %d has is now going to deliver at home', individual_id)

        # Check only women delivering at home pass through this event
        assert mni[individual_id]['delivery_setting'] == 'home_birth'

        # Condition the event on women being alive
        if df.at[individual_id, 'is_alive']:
            logger.info('%s|home_birth|%s', self.sim.date,
                        {'mother_id': individual_id})

            # ===================================  APPLICATION OF COMPLICATIONS =======================================

            # Using the complication_application function we move through each complication in turn determining if a
            # woman will experience any of these if she has delivered at home, and storing the risk so it can be
            # modified by skilled birth attendance for women delivering in a facility
            # Here labour_stage 'ip' means intrapartum

            self.module.set_home_birth_complications_intrapartum(individual_id, labour_stage='ip', complication=
                                                                 'obstructed_labour')
            self.module.set_home_birth_complications_intrapartum(individual_id, labour_stage='ip', complication=
                                                                 'antepartum_haem')
            self.module.set_home_birth_complications_intrapartum(individual_id, labour_stage='ip', complication=
                                                                 'sepsis')
            self.module.set_home_birth_complications_intrapartum(individual_id, labour_stage='ip', complication=
                                                                 'eclampsia')
            self.module.set_home_birth_complications_intrapartum(individual_id, labour_stage='ip', complication=
                                                                 'uterine_rupture')

            # ==============================  CARE SEEKING FOLLOWING COMPLICATIONS ====================================

            # We use a logistic regression equation, stored in the linear model, to determine if women will seek care
            # for delivery if they have developed a complication at a home birth
            facility_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(self.module,
                                                                                              person_id=individual_id)

            if df.at[individual_id, 'la_obstructed_labour'] \
                or df.at[individual_id, 'la_antepartum_haem'] \
                or df.at[individual_id, 'la_sepsis']\
                or df.at[individual_id, 'la_eclampsia'] \
                or df.at[individual_id, 'la_uterine_rupture']:
                if self.module.eval(params['la_labour_equations']['care_seeking_for_complication'], individual_id):
                    mni[individual_id]['sought_care_for_complication'] = True
                    logger.debug('mother %d will now seek care for a complication that has developed during labour '
                                 'on date %s', individual_id, self.sim.date)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

            # TODO: this equation only determines if the woman seeks care, not at what level
            # TODO: at what point do these women enter the HSI, or should there maybe be a different one? And what
            #  about facility level selection


class BirthEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by LabourOnsetEvent. For women who survived labour, the appropriate
    variables are reset/updated and the function do_birth is executed. This event schedules PostPartumLabourEvent for
    those women who have survived"""

    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        df = self.sim.population.props
        person = df.loc[mother_id]
        logger.info('mother %d at birth event on date %s', mother_id, self.sim.date)

        # Check the correct amount of time has passed between labour onset and birth event
        assert (self.sim.date - df.at[mother_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')

        # If the mother is alive and still pregnant we generate a  child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications (intrapartum stillbirths till
        # trigger births for monitoring purposes)
        if person.is_alive and person.is_pregnant:
            logger.info('@@@@ A Birth is now occuring, to mother %d', mother_id)
            self.sim.do_birth(mother_id)
            logger.debug('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id),
                                    self.sim.date)

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~person.is_alive and ~person.la_intrapartum_still_birth and person.la_maternal_death_in_labour:
            logger.debug('@@@@ A Birth is now occuring, to mother %d who died in childbirth but her child survived',
                         mother_id)
            self.sim.do_birth(mother_id)


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):
    """This is PostpartumLabour event. It is scheduled by BirthEvent immediately following birth. This event applies the
    probability that women delivering at home will experience complications, makes the appropriate changes to the data
    frame. This event schedules the PostPartumDeathEvent (4 days post this event)and the DisabilityResetEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')

        # Event should only run if woman is still alive
        if df.at[individual_id, 'is_alive']:

            # Here we use the set_home_birth_complications function to determine if a women who has survived labour at
            # home will experience any further complications
            if mni[individual_id]['delivery_setting'] == 'home_birth':

                # Here labour_stage 'pp' means postpartum
                self.module.set_home_birth_complications_postpartum(individual_id, labour_stage='pp',
                                                                    complication='postpartum_haem')
                if ~df.at[individual_id, 'la_sepsis']:
                    # Women who already have sepsis cannot develop it again immediately after birth
                    self.module.set_home_birth_complications_postpartum(individual_id, labour_stage='pp',
                                                                        complication='sepsis')
                    # Women who already have already developed eclampsia cannot develop it again immediately after birth
                if ~df.at[individual_id, 'la_eclampsia']:
                    self.module.set_home_birth_complications_postpartum(individual_id, labour_stage='pp',
                                                                        complication='eclampsia')

            event = HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1(self.module, person_id=individual_id)

            # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[individual_id]['delivery_setting'] != 'home_birth':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id, self.sim.date)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

            # If a woman has delivered at home and develops a complication, we now determine if she will seek care using
            # an equation from the linear model
            # TODO: this, again, will schedule the event a second time for anyone who develops sepsis or eclampsia ip
            elif mni[individual_id]['delivery_setting'] == 'home_birth'and \
                (df.at[individual_id, 'la_sepsis_postpartum'] or df.at[individual_id, 'la_eclampsia_postpartum']
                 or df.at[individual_id, 'la_postpartum_haem']):
                    if self.module.eval(params['la_labour_equations']['care_seeking_for_complication'], individual_id):
                        mni[individual_id]['sought_care_for_complication'] = True
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))
                    logger.debug('mother %d will now seek care for a complication that has developed following labour '
                                 'on date %s', individual_id, self.sim.date)

            # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
            # complications may experience death
            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id), self.sim.date+ DateOffset(days=4))
            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                        self.sim.date + DateOffset(days=4))  # Date offset to allow for interventions

            # Here we schedule women to an event which resets 'daly' disability associated with delivery complications
            self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id), self.sim.date +
                                    DateOffset(months=1))


class LabourDeathEvent (Event, IndividualScopeEventMixin):
    """This is the LabourDeathEvent. It is scheduled by the LabourOnsetEvent for all women who go through labour. This
    event determines if women who have experienced complications in labour will die or experience an intrapartum
    stillbirth."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(3, unit='D')

        # For women who have experienced a complication we use the linear model to calculate their individual risk of
        # death/stillbirth. Where women have received treatment, their risk of death is modulated by the treatment
        # effect
        def set_maternal_death_status_intrapartum(cause):
            if self.module.eval(params['la_labour_equations'][f'{cause}_death'], individual_id):
                logger.debug(f'{cause} has contributed to person %d death during labour', individual_id)

                mni[individual_id]['death_in_labour'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(cause)
                df.at[individual_id, 'la_maternal_death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date

                # then if she does die, we determine if the child will still survive
            if self.module.eval(params['la_labour_equations'][f'{cause}_stillbirth'], individual_id):
                logger.debug(f'person %d has experienced a still birth following {cause} in labour')

                df.at[individual_id, 'la_intrapartum_still_birth'] = True
                df.at[individual_id, 'ps_previous_stillbirth'] = True

        # First we determine if the mother will die due to her complication. Women cannot die of obstructed labour
        # directly, but can die of complications for which obstructed labour is a risk (i.e. sepsis and uterine rupture)

        if df.at[individual_id, 'la_eclampsia']:
            set_maternal_death_status_intrapartum(cause='eclampsia')

        if df.at[individual_id, 'la_antepartum_haem']:
            set_maternal_death_status_intrapartum(cause='antepartum_haem')

        if df.at[individual_id, 'la_sepsis']:
            set_maternal_death_status_intrapartum(cause='sepsis')

        if df.at[individual_id, 'la_uterine_rupture']:
            set_maternal_death_status_intrapartum(cause='uterine_rupture')

            # TODO: Will we apply a reduced CFR in the instance of unsuccessful interventions

        # Schedule death for women who die in labour
        if mni[individual_id]['death_in_labour']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)
            # TODO: amend cause= 'labour_' + [str(cause) + '_' for cause in list(mni[individual_id]
            #  [cause_of_death_in_labour]

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
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.debug('mother %d at PPD at date %s', individual_id, self.sim.date)

        # Check the correct amount of time has passed between labour onset and PostPartumDeathEvent
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(8, unit='D')

        # We apply the same structure as with the LabourDeathEvent to women who experience postpartum complications

        def set_maternal_death_status_postpartum(cause):
            if self.module.eval(params['la_labour_equations'][f'{cause}_pp_death'], individual_id):
                logger.debug(f'{cause} has contributed to person %d death following labour', individual_id)

                mni[individual_id]['death_postpartum'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
                df.at[individual_id, 'la_maternal_death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date

        # Check the woman is currently alive
        if df.at[individual_id, 'is_alive']:
            if mni[individual_id]['eclampsia_pp']:
                set_maternal_death_status_postpartum(cause='eclampsia')

            if df.at[individual_id, 'la_postpartum_haem']:
                set_maternal_death_status_postpartum(cause='postpartum_haem')

            if mni[individual_id]['sepsis_pp']:
                set_maternal_death_status_postpartum(cause='sepsis')

            if mni[individual_id]['death_postpartum']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='postpartum labour'), self.sim.date)

        # TODO: amend cause= 'labour_' + [str(cause) + '_' for cause in list(mni[individual_id]
        #  [cause_of_death_in_labour]

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
                # TODO: Consider how best to deal with complications that are long lasting.

                logger.debug('%s|labour_complications|%s', self.sim.date,
                             {'person_id': individual_id,
                              'labour_profile': mni[individual_id]})

            # End the period of current labour
            df.at[individual_id, 'la_currently_in_labour'] = False


class DisabilityResetEvent (Event, IndividualScopeEventMixin):
    """This is the DisabilityResetEvent. It is scheduled by the PostPartumLabourEvent. This event resets a woman's
    disability properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # TODO: Confirm this allows enough time for model to count DALYs before resetting

        # Here we turn off all the properties which are used to count DALYs
        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their disability status reset', individual_id)

            df.at[individual_id, 'la_sepsis_disab'] = False
            df.at[individual_id, 'la_obstructed_labour_disab'] = False
            df.at[individual_id, 'la_uterine_rupture_disab'] = False
            df.at[individual_id, 'la_eclampsia_disab'] = False
            df.at[individual_id, 'la_maternal_haem_severe_disab'] = False
            df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = False


class DiseaseResetEvent (Event, IndividualScopeEventMixin):
    """This is the DiseaseResetEvent. It is scheduled by the PostPartumDeathEvent. This event resets a woman's
    disease properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        # This event ensures that for women who have survived delivery but have suffered a complication in which
        # treatment was unsuccessful have their diseases variables reset

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

            del mni[individual_id]

# ======================================================================================================================
# ================================ HEALTH SYSTEM INTERACTION EVENTS ====================================================
# ======================================================================================================================


class HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabourFacilityLevel1. This event is scheduled by the LabourOnset
    Event. This event manages initial care around the time of delivery including prophylactic interventions (i.e. clean
    birth practices) for women presenting at Level 1 of the health system for delivery care. This event uses a womans
    stored risk of complications, which may be manipulated by treatment effects to determines if they will experience
    a complication during their labour in hospital. It is responsible for scheduling treatment HSIs for those
    complications."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Mother %d has presented to a health '
                    'facility on date %s following the onset of her labour', person_id, self.sim.date)

        # ======================================= LOGGING DELIVERY ================================================
        # Women who developed a complication at home, then presented to a facility for delivery, are counted as
        # facility deliveries
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'
            # TODO: awaiting choice of facility type to be included in complication care seeking equation

        if mni[person_id]['delivery_setting'] == 'health_centre':
            logger.info('%s|health_centre_delivery|%s', self.sim.date,
                        {'person_id': person_id})

        elif mni[person_id]['delivery_setting'] == 'hospital':
            logger.info('%s|hospital_delivery|%s', self.sim.date,
                        {'person_id': person_id})

        # ============================== STORING COMPLICATION RISKS IN MNI ========================================
        # We calculate a woman's individual risk of each complication at the start of the HSI, and store this in her
        # MNI dictionary. This allows the risk to be manipulated by prophylactic interventions
        # todo: We dont modify the risk of ALL complications during this HSI, therefore not all complication functions
        #  need to fire

        # Women who develop a complication and seek care will already have had their risk calculated, so they are
        # excluded here
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='obstructed_labour',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='eclampsia',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='antepartum_haem',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='sepsis',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='uterine_rupture',
                                                                      labour_stage='ip')

        # ============================== AVAILABILITY SKILLED BIRTH ATTENDANTS =====================================
        # On presentation to the facility, we use the squeeze factor to determine if this woman will receive delivery
        # care from a health care professional, or if she will delivered unassisted in a facility
        if squeeze_factor > params['squeeze_factor_threshold_delivery_attendance']:
            mni[person_id]['delivery_attended'] = False
            logger.debug('mother %d is delivering without assistance at a level 1 health facility', person_id)
            logger.info(('%s|unattended_facility_delivery|%s', self.sim.date,
                         {'age': df.at[person_id, 'age_years'],
                          'person_id': person_id}))
        else:
            mni[person_id]['delivery_attended'] = True
            logger.debug('mother %d is delivering with assistance at a level 1 health facility', person_id)

        # Run checks that key facility properties in the mni have been set
        assert mni[person_id]['delivery_attended'] is not None
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        # ===================================== PROPHYLACTIC CARE ===================================================
        # The following function manages the consumables and administration of prophylactic interventions in labour

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # TODO: Confirm how to ensure that only the consumables NEEDED are those which are logged (whether or not they
        #  are available) as required consumables vary from woman to woman based on presenting history
        # TODO: Define refined consumable packages within the module

        def prophylactic_labour_interventions():
            #  We define all the possible consumables that could be required
            # Baseline consumables for delivery...
            pkg_code_uncomplicated_delivery = pd.unique(consumables.loc[consumables[
                                                                    'Intervention_Pkg'] == 'Vaginal delivery - skilled '
                                                                                           'attendance',
                                                                    'Intervention_Pkg_Code'])[0]

            # A clean delivery kit to reduce probability of maternal and newborn sepsis...
            pkg_code_clean_delivery_kit = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                                    'Clean practices and immediate essential newborn '
                                                                    'care (in facility)', 'Intervention_Pkg_Code'])[0]

            # Antibiotics for women whose membranes have ruptured prematurely...
            item_code_abx_prom = pd.unique(consumables.loc[consumables['Items'] == 'Benzylpenicillin 1g (1MU),'
                                                                                   ' PFR_Each_CMST', 'Item_Code'])[0]
            pkg_code_pprom = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM',
                                                       'Intervention_Pkg_Code'])[0]
            # n.b 2 additional IV abx not in guidelines

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

            # we check if consumables are available then administered prophylactic interventions accordingly:
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_attended_delivery,
                to_log=True)
            # Clean birth practices...
            # We query if the consumables specific to each intervention are available...
            if outcome_of_request_for_consumables['Intervention_Package_Code'][
                pkg_code_clean_delivery_kit]:
                # And apply the effect of these intervention on the mother and newborns risk of sepsis
                mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] * \
                    params['rr_maternal_sepsis_clean_delivery']
                mni[person_id]['risk_newborn_sepsis'] = mni[person_id]['risk_newborn_sepsis'] * \
                    params['rr_newborn_sepsis_clean_delivery']

                logger.debug('This facility has delivery kits available and have been used for mother %d delivery.',
                             person_id)

                # If consumables are not available, the intervention is not delivered
            else:
                logger.debug('This facility has no delivery kits.')

            # Prophylactic antibiotics for premature rupture of membranes in term deliveries...
            if mni[person_id]['labour_state'] == 'term_labour' and df.at[person_id,
                                                                         'ps_premature_rupture_of_membranes']:
                if outcome_of_request_for_consumables['Item_Code'][item_code_abx_prom]:
                    mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] *\
                                                       params['rr_sepsis_post_abx_prom']

                    logger.debug('This facility has provided antibiotics for mother %d who is a risk of sepsis due '
                                 'to PROM.', person_id)
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')

            # Prophylactic antibiotics for premature rupture of membranes in preterm deliveries...
            if (mni[person_id]['labour_state'] == 'early_preterm_labour' or mni[person_id]['labour_state'] ==
               'late_preterm_labour') and df.at[person_id, 'ps_premature_rupture_of_membranes']:
                if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pprom]:
                    mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] * \
                                                       params['rr_sepsis_post_abx_pprom']

                    logger.debug('This facility has provided antibiotics for mother %d who is a risk of sepsis due '
                                 'to PPROM.', person_id)
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')
                    # TODO: Review treatment guidelines for PPROM treatment

            # Prophylactic steroids for women who have gone into preterm labour...
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or mni[person_id]['labour_state'] == \
               'late_preterm_labour':
                if outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_betamethasone] and \
                   outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_dexamethasone]:
                    mni[person_id]['corticosteroids_given'] = True
                    logger.debug('This facility has provided corticosteroids for mother %d who is in preterm labour',
                                 person_id)
                else:
                    logger.debug('This facility has no steroids for women in preterm labour.')

            # Prophylactic antibiotics for women who have gone into preterm labour...
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or mni[person_id]['labour_state'] == \
               'late_preterm_labour':
                if outcome_of_request_for_consumables['Item_Code'][item_code_antibiotics_gbs_proph]:
                    mni[person_id]['risk_newborn_sepsis'] = mni[person_id]['risk_newborn_sepsis'] * \
                                                            params['rr_newborn_sepsis_proph_abx']
                    logger.debug('This facility has provided antibiotics for mother %d whose baby is a risk of '
                                 'Group B strep', person_id)
                else:
                    logger.debug('This facility has no antibiotics for group B strep prophylaxis.')

        # If this womans delivery is attended, the function will run. However women presenting for delivery following a
        # complication do not benefit from prophylaxis as the risk of complications has already been applied
        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            prophylactic_labour_interventions()
        else:
            # Otherwise she receives no benefit of prophylaxis
            logger.debug('mother %d received no prophylaxis as she is delivering unattended', person_id)

        # ================================= PROPHYLACTIC MANAGEMENT PRE-ECLAMPSIA  ==============================
        # As women with severe pre-eclampsia are the most at risk group of women to develop eclampsia, they should
        # receive magnesium to reduce the probability of seizures
        def assessment_and_treatment_of_severe_pre_eclampsia(facility_type):
            pkg_code_severe_preeclampsia = \
            pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                      'Intervention_Pkg_Code'])[0]

            consumables_needed_spe = {'Intervention_Package_Code': {pkg_code_severe_preeclampsia: 1},
                                      'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_spe)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                           f'assess_severe_pe_{facility_type}',
                                                                           hsi_event=self):
                if outcome_of_request_for_consumables_sep:
                    df.at[person_id, 'la_severe_pre_eclampsia_treatment'] = True
                    logger.debug('mother %d has has their severe pre-eclampsia identified during delivery. As '
                                 'consumables are available they will receive treatment', person_id)

            elif df.at[person_id, 'ps_severe_pre_eclamp']:
                logger.debug('mother %d has not had their severe pre-eclampsia identified during delivery and will '
                                 'not be treated', person_id)

        # We assume that only women who are having an attended delivery will have their severe pre-eclampsia identified
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_treatment_of_severe_pre_eclampsia('hc')
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'hospital':
            assessment_and_treatment_of_severe_pre_eclampsia('hp')

    # ===================================== APPLYING COMPLICATION INCIDENCE ===========================================
        # Following administration of prophylaxis (for attended deliveries) we use risk values stored in the MNI to
        # assess if this woman will develop any complications
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='obstructed_labour',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='eclampsia',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='antepartum_haem',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='sepsis',
                                                                            labour_stage='ip')

            # n.b. we do not apply the risk of uterine rupture due to the causal link between obstructed labour and
            # uterine rupture. We want interventions for obstructed labour to reduce the risk of uterine rupture

    # ======================================= COMPLICATION MANAGEMENT =================================================
        # Management of each complication is housed within its own function which follows a similar pattern
        # a.) required consumables are defined
        # b.) The woman is assessed for a complication using the dx_test function. Specificity of assesment varies
        #     between facility level
        # c.) If she has the complication and it is correctly identified by staff they check consumables are
        #     available
        # d.) If the consumables are available- she will receive treatment
        # If the woman is delivering unattended, we assume any treatment she receives is delayed, delayed treatment is
        # less effective meaning risk of death is increased

        # In the instance that treatment cannot be delivered at the facility type a woman has delivered at (i.e.
        # health centres do not perform caesarean sections) she is assessed for referral and sent to another
        # facility

        # -----------------------------------------Obstructed Labour: -------------------------------------------------
        def assessment_and_treatment_of_obstructed_labour(facility_type):
            # Required consumables are defined
            pkg_code_obstructed_labour = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                                   'Management of obstructed labour',
                                                                   'Intervention_Pkg_Code'])[0]

            item_code_forceps = pd.unique(consumables.loc[consumables['Items'] == 'Forceps, obstetric', 'Item_Code'])[0]
            item_code_vacuum = pd.unique(consumables.loc[consumables['Items'] == 'Vacuum, obstetric', 'Item_Code'])[0]

            consumables_obstructed_labour = {'Intervention_Package_Code': {pkg_code_obstructed_labour: 1},
                                             'Item_Code': {item_code_forceps: 1, item_code_vacuum: 1}}

            # todo: if we're combining the effectiveness then we want forceps OR vacuum
            # TODO: as ol pkg is very comprehensive, may be a bad idea to condition on this

            outcome_of_request_for_consumables_ol = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_obstructed_labour, to_log=True)

            # We determine if this complication will be identified at this facility level
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                       f'assess_obstructed_labour_{facility_type}',
                                                                       hsi_event=self):
                if outcome_of_request_for_consumables_ol:
                    # If the consumables are available, treatment occurs. It is prompt or delayed depending on if the
                    # delivery is attended or not
                    logger.debug('mother %d has had her obstructed labour identified during delivery. Staff will '
                                 'attempt an assisted vaginal delivery as the equipment is available', person_id)
                    treatment_success = params['prob_successful_assisted_vaginal_delivery'] > self.module.rng.\
                        random_sample()

                    if treatment_success and mni[person_id]['delivery_attended']:
                        df.at[person_id, 'la_obstructed_labour_treatment'] = 'prompt_treatment'
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    elif treatment_success and ~mni[person_id]['delivery_attended']:
                        df.at[person_id, 'la_obstructed_labour_treatment'] = 'delayed_treatment'
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    # If treatment is unsuccessful, the woman will be referred for additional treatment
                    else:
                        logger.debug('Following a failed assisted vaginal delivery other %d will need additional '
                                     'treatment', person_id)

                        if mni[person_id]['delivery_attended']:
                            mni[person_id]['refer_for_cs'] = 'prompt_referral'
                        else:
                            mni[person_id]['refer_for_cs'] = 'delayed_referral'
                    # TODO: if a woman is referred for CS should this mean she is unable to experience uterine rutpure?

            # If treatment is not not provided, as need is not identified, risk of death and stillbirth will be higher
            # in the linear model
            elif df.at[person_id, 'la_obstructed_labour']:
                logger.debug('mother %d has not had their obstructed labour identified during delivery and will not'
                             'be treated', person_id)

        # We use variable sensitivity of the assessment of these complications by facility type (to mimic
        # gradient in quality)
        if mni[person_id]['delivery_setting'] == 'health_centre':
                assessment_and_treatment_of_obstructed_labour('hc')
        if mni[person_id]['delivery_setting'] == 'hospital':
                assessment_and_treatment_of_obstructed_labour('hp')

    # ---------------------------------------------- Maternal Sepsis: -------------------------------------------------
        # We follow a similar pattern for treatment of sepsis
        def assessment_and_treatment_of_maternal_sepsis(facility_type, labour_stage):
            pkg_code_sepsis = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case'
                                                                                           ' management',
                                                        'Intervention_Pkg_Code'])[0]

            consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_sepsis)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_sepsis_{facility_type}_'
                                                                       f'{labour_stage}',
                                                                       hsi_event=self):
                if outcome_of_request_for_consumables_sep:
                    logger.debug('mother %d has has their sepsis identified during delivery. As consumables are '
                                 'available they will receive treatment', person_id)
                    if mni[person_id]['delivery_attended']:
                        df.at[person_id, 'la_sepsis_treatment'] = 'prompt_treatment'
                    else:
                        df.at[person_id, 'la_sepsis_treatment'] = 'delayed_treatment'
            elif df.at[person_id, 'la_sepsis']:
                logger.debug('mother %d has not had their sepsis identified during delivery and will not be treated',
                             person_id)

        if mni[person_id]['delivery_setting'] == 'health_centre':
                assessment_and_treatment_of_maternal_sepsis('hc', 'ip')
        if mni[person_id]['delivery_setting'] == 'hospital':
                assessment_and_treatment_of_maternal_sepsis('hp', 'ip')

    # --------------------------------------------------- Maternal Hypertension ---------------------------------------
            # TODO: are we gunna group mild pre-eclampsia, gest htn and chronic htn all together?

        # and for hypertension
        def assessment_and_treatment_of_hypertension(facility_type):
            pkg_code_severe_hypertension = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Management of '
                                                                                                        'eclampsia',
                                                                     'Intervention_Pkg_Code'])[0]

            consumables_needed_htn = {'Intervention_Package_Code': {pkg_code_severe_hypertension: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_htn = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_htn)
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                       f'assess_hypertension_{facility_type}',
                                                                       hsi_event=self):
                if outcome_of_request_for_consumables_htn:
                    df.at[person_id, 'la_maternal_hypertension_treatment'] = True
                    logger.debug('mother %d has has their hypertension identified during delivery. As '
                                 'consumables are available they will receive treatment', person_id)

            elif df.at[person_id, 'ps_gestational_htn']:
                    logger.debug('mother %d has not had their hypertension identified during delivery and will '
                                 'not be treated', person_id)

        # However we assume that only women who are having an attended delivery will have their hypertension identified
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_treatment_of_hypertension('hc')
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'hospital':
            assessment_and_treatment_of_hypertension('hp')

    # ------------------------------------------------ ECLAMPSIA  ------------------------------------------------------

        # Because eclampsia presents and easily recognisable seizures we do not use the dx_test function and assume all
        # women have equal likelihood of treatment irrespective of facility level or delivery attendance
        def assessment_and_treatment_of_eclampsia():
            pkg_code_eclampsia = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                                           'Intervention_Pkg_Code'])[0]

            consumables_needed_eclampsia = {'Intervention_Package_Code': {pkg_code_eclampsia: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_ec = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_eclampsia)

            if outcome_of_request_for_consumables_ec:
                df.at[person_id, 'la_eclampsia_treatment'] = True
                logger.debug('mother %d has has their eclampsia identified during delivery. As '
                             'consumables are available they will receive treatment', person_id)
            elif df.at[person_id, 'la_eclampsia']:
                logger.debug('mother %d has not had their eclampsia identified during delivery and will '
                             'not be treated', person_id)

        if df.at[person_id, 'la_eclampsia']:
            assessment_and_treatment_of_eclampsia()

    # ------------------------------------------------ ANTEPARTUM HAEMORRHAGE -----------------------------------------
        # Antepartum haemorrhage can only be treated by delivery in our model (as both causes are placental). Here the
        # dx_test in a health centre has its sensitivity set to reflect likelihood this woman will be correctly referred
        # (as she will need another facility). The dx_test in a hospital has its sensitivity set to reflect likelihood
        # of treatment within the same facility type, as surgical services are available

        def assessment_and_plan_for_referral_antepartum_haemorrhage(facility_type, treatment_or_referral):
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test\
                    (dx_tests_to_run=f'assess_for_{treatment_or_referral}_aph_{facility_type}',
                     hsi_event=self):
                if mni[person_id]['delivery_attended']:
                    mni[person_id]['referred_for_cs'] = 'prompt_referral'
                    mni[person_id]['referred_for_blood'] = 'prompt_referral'
                else:
                    mni[person_id]['referred_for_cs'] = 'delayed_referral'
                    mni[person_id]['referred_for_blood'] = 'delayed_referral'
                logger.debug('mother %d has has their antepartum haemorrhage identified during delivery. They will now '
                             'be referred for additional treatment', person_id)
            elif df.at[person_id, 'la_antepartum_haem']:
                logger.debug('mother %d has not had their antepartum haemorrhage identified during delivery and will'
                             'not be referred for treatment', person_id)

        # Whilst we assume a difference in sensitivity of assessment between health centres and hospitals, we assume
        # women who are delivering unattended have the same likelihood of treatment due to the presentation of this
        # condition
        if mni[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_plan_for_referral_antepartum_haemorrhage('hc', 'referral')
        if mni[person_id]['delivery_setting'] == 'hospital':
            assessment_and_plan_for_referral_antepartum_haemorrhage('hp', 'treatment')

    # ------------------------------------------------ UTERINE RUPTURE ------------------------------------------------
        # We now calculate and apply the risk of uterine rupture
        self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='uterine_rupture',
                                                                        labour_stage='ip')

        # Similarly treatment of uterine rupture requires comprehensive services, therefore the dx test is for assemenet
        # for referral
        def assessment_and_plan_for_referral_uterine_rupture(facility_type, treatment_or_referral):
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test\
                    (dx_tests_to_run=f'assess_for_{treatment_or_referral}_uterine_rupture_{facility_type}',
                     hsi_event=self):
                if mni[person_id]['delivery_attended']:
                    mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                    mni[person_id]['referred_for_cs'] = 'prompt_referral'
                else:
                    mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                    mni[person_id]['referred_for_cs'] = 'delayed_referral'
                # ? blood?
                logger.debug('mother %d has has their uterine rupture identified during delivery. They will now be '
                             'referred for additional treatment', person_id)
            elif df.at[person_id, 'la_uterine_rupture']:
                logger.debug('mother %d has not had their uterine_rupture identified during delivery and will not be '
                             'referred for treatment', person_id)

        if mni[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_plan_for_referral_uterine_rupture('hc', 'referral')
        if mni[person_id]['delivery_setting'] == 'hospital':
            assessment_and_plan_for_referral_uterine_rupture('hp', 'treatment')

    #
    # ============================================== REFERRAL =========================================================
        # Finally we send any women who require additional treatment to the following HSIs

        if mni[person_id]['referred_for_cs'] != 'none':
            caesarean_section = HSI_Labour_CaesareanSectionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(caesarean_section,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # If a this woman has experienced a complication the appointment footprint is changed from normal to complicated
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        if df.at[person_id, 'la_sepsis'] \
            or df.at[person_id, 'la_antepartum_haem'] \
            or df.at[person_id,'la_uterine_rupture'] \
            or df.at[person_id, 'la_eclampsia']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery']  # is this right?

        return actual_appt_footprint

    # In the event that the health system doesnt have the capacity to run this event, the woman will deliver at home
    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she will now '
                     'deliver at home', person_id, self.sim.date)

        logger.info('%s|homebirth_high_squeeze|%s', self.sim.date,
                    {'person_id': person_id})

        mni[person_id]['delivery_setting'] = 'home_birth'
        self.sim.schedule_event(LabourAtHomeEvent(self, person_id), self.sim.date)


class HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabourFacilityLevel2. This event is scheduled by the LabourOnset
    Event. This event manages initial care around the time of delivery including prophylactic interventions (i.e. clean
    birth practices) for women presenting at Level 2 of the health system for delivery care. This event uses a womans
    stored risk of complications, which may be manipulated by treatment effects to determines if they will experience
    a complication during their labour in hospital. It is responsible for scheduling treatment HSIs for those
    complications."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabourFacilityLevel2'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel2: Mother %d has presented to '
                    'a health facility on date %s following the onset of her labour', person_id, self.sim.date)

        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'hospital'

        logger.info('%s|hospital_delivery|%s', self.sim.date,
                    {'person_id': person_id})

        # The structure of this event mirrors HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1. There is no
        # differentiation between facility types as facility level 2 is exclusively hospital based.

        # ============================== STORING COMPLICATION RISKS IN MNI ========================================
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='antepartum_haem',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='sepsis',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='uterine_rupture',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='obstructed_labour',
                                                                      labour_stage='ip')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='eclampsia',
                                                                      labour_stage='ip')

        if squeeze_factor > params['squeeze_factor_threshold_delivery_attendance']:
            mni[person_id]['delivery_attended'] = False
            logger.debug('mother %d is delivering without assistance at a level 1 health facility', person_id)
            logger.info(('%s|unattended_facility_delivery|%s', self.sim.date,
                         {'age': df.at[person_id, 'age_years'],
                          'person_id': person_id}))
        else:
            mni[person_id]['delivery_attended'] = True
            logger.debug('mother %d is delivering with assistance at a level 1 health facility', person_id)

        assert mni[person_id]['delivery_attended'] is not None
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            self.module.prophylactic_labour_interventions()
        else:
            logger.debug('mother %d received no prophylaxis as she is delivering unattended', person_id)

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='obstructed_labour',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='eclampsia',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='antepartum_haem',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='sepsis',
                                                                            labour_stage='ip')
            self.module.set_complications_during_facility_birth_intrapartum(person_id, complication='uterine_rupture',
                                                                            labour_stage='ip')

        # ===================================== COMPLICATION ASSESSMENT ===========================================
        self.module.assessment_and_treatment_of_obstructed_labour('hp')

        self.module.assessment_and_treatment_of_maternal_sepsis('hp', 'ip')

        if mni[person_id]['delivery_attended']:
            self.module.assessment_and_treatment_of_hypertension('hp')

        if mni[person_id]['delivery_attended']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia('hp')

        if df.at[person_id, 'la_eclampsia']:
            self.module.assessment_and_treatment_of_eclampsia()

        self.module.assessment_and_plan_for_referral_antepartum_haemorrhage('hp', 'treatment')

        self.module.assessment_and_plan_for_referral_uterine_rupture('hp', 'treatment')

    # ============================================== REFERRAL =========================================================
        # TODO: should this event only schedule an the level 2 events?
        if mni[person_id]['referred_for_cs'] != 'none':
            caesarean_section = HSI_Labour_CaesareanSectionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(caesarean_section,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        if df.at[person_id, 'la_sepsis'] \
            or df.at[person_id, 'la_antepartum_haem'] \
            or df.at[person_id,'la_uterine_rupture'] \
            or df.at[person_id, 'la_eclampsia']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery']  # is this right?

        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she will now '
                     'deliver at home', person_id, self.sim.date)

        logger.info('%s|homebirth_high_squeeze|%s', self.sim.date,
                    {'person_id': person_id})

        mni[person_id]['delivery_setting'] = 'home_birth'
        self.sim.schedule_event(LabourAtHomeEvent(self, person_id), self.sim.date)


class HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1. This event is scheduled by the
    PostpartumLabourEvent . This event manages initial care around the time of delivery including prophylactic
    interventions (i.e. clean birth practices) for women presenting at Level 1 of the health system for delivery care.
    This event uses a womans stored risk of complications, which may be manipulated by treatment effects to determines
    if they will experience a complication during their labour in hospital. It is responsible for scheduling treatment
    HSIs for those complications.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1: Providing skilled attendance '
                    'following birth for person %d', person_id)

        # Although we change the delivery setting variable to 'facility_delivery' we do not include women who present
        # for care following birth, due to complications, as facility deliveries
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # ================================= STORING COMPLICATION RISKS IN MNI ==========================================
        # As with  HSI_Labour_PresentsForSkilledAttendanceInLabour we start by calculating and storing each womans
        # individual risk of each complication

        if ~mni[person_id]['sought_care_for_complication']:
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='eclampsia',
                                                                      labour_stage='pp')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='postpartum_haem',
                                                                      labour_stage='pp')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='sepsis',
                                                                      labour_stage='pp')

        # ------------------ Active Management of the third stage of labour -------------------------------------------
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        def active_management_of_the_third_stage_of_labour():
            pkg_code_am = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] ==
                                                    'Active management of the 3rd stage of labour',
                                                    'Intervention_Pkg_Code'])[0]

            consumables_needed = {
                'Intervention_Package_Code': {pkg_code_am: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
            # labour (additional oxytocin, uterine massage and controlled cord traction)
            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_am]:
                    logger.debug('pkg_code_am is available, so use it.')
                    mni[person_id]['risk_pp_postpartum_haem'] = mni[person_id]['risk_pp_postpartum_haem'] * \
                        params['rr_pph_amtsl']
            else:
                logger.debug('mother %d did not receive active management of the third stage of labour as she delivered'
                             'without assistance', person_id)

        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            active_management_of_the_third_stage_of_labour()
        else:
            logger.debug('mother %d did not receive active management of the third stage of labour due to resource '
                         'constraints')

    # ===================================== APPLYING COMPLICATION INCIDENCE ===========================================
    # Again we use the mothers individual risk of each complication to determine if she will experience any
    # complications using the set_complications_during_facility_birth function.
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.set_complications_during_facility_birth_postpartum(person_id, complication='eclampsia',
                                                                           labour_stage='pp')
            self.module.set_complications_during_facility_birth_postpartum(person_id, complication='postpartum_haem',
                                                                           labour_stage='pp')

            # Only women who haven't already developed sepsis are able to become septic at this point
            if ~df.at[person_id, 'la_sepsis']:
                self.module.set_complications_during_facility_birth_postpartum(person_id, complication='sepsis',
                                                                               labour_stage='pp')

    # ======================================= COMPLICATION MANAGEMENT =================================================

    # ------------------------------------------------ SEPSIS  ------------------------------------------------------
        # The same function is applied for the assessment and treatment of postpartum sepsis as with
        # HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel...
        def assessment_and_treatment_of_maternal_sepsis(facility_type, labour_stage):
            pkg_code_sepsis = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case'
                                                                                           ' management',
                                                        'Intervention_Pkg_Code'])[0]

            consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_sepsis)
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_sepsis_{facility_type}_'
                                                                       f'{labour_stage}',
                                                                       hsi_event=self):
                if outcome_of_request_for_consumables_sep:
                    logger.debug('mother %d has has their sepsis identified following delivery. As consumables are '
                                 'available they will receive treatment', person_id)
                    if mni[person_id]['delivery_attended']:
                        df.at[person_id, 'la_sepsis_treatment'] = 'prompt_treatment'
                    else:
                        df.at[person_id, 'la_sepsis_treatment'] = 'delayed_treatment'

            elif df.at[person_id, 'la_sepsis_postpartum']:
                logger.debug('mother %d has not had their sepsis identified during delivery and will not be treated',
                             person_id)

        # todo: will this not double run this event on women who are already septic?
        if mni[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_treatment_of_maternal_sepsis('hc', 'pp')
        if mni[person_id]['delivery_setting'] == 'hospital':
            assessment_and_treatment_of_maternal_sepsis('hp', 'pp')

    # ------------------------------------------------ ECLAMPSIA  -----------------------------------------------------
        # The same function is applied for the assessment and treatment of postpartum eclampsia as with
        # HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel...
        def assessment_and_treatment_of_eclampsia():
            pkg_code_eclampsia = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                                           'Intervention_Pkg_Code'])[0]

            consumables_needed_eclampsia = {'Intervention_Package_Code': {pkg_code_eclampsia: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_ec = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_eclampsia)

            if outcome_of_request_for_consumables_ec:
                df.at[person_id, 'la_eclampsia_treatment'] = True
                logger.debug('mother %d has has their eclampsia identified during delivery. As '
                             'consumables are available they will receive treatment', person_id)
            elif df.at[person_id, 'la_eclampsia_postpartum']:
                logger.debug('mother %d has not had their eclampsia identified during delivery and will '
                             'not be treated', person_id)

        if df.at[person_id, 'la_eclampsia_postpartum']:
            assessment_and_treatment_of_eclampsia()

        # ----------------------------------- POSTPARTUM HAEMORRHAGE --------------------------------------------------
        # Whilst awaiting to refine the consumables required for the management of postpartum haemorrhage we are using
        # the consumable package defined below.
        # TODO: refine these consumables considerably

        pkg_code_pph = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum '
                                                                                    'hemorrhage', 'Intervention_Pkg_'
                                                                                                  'Code'])[0]

        consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_pph)

        # We assume apply a delayed treatment effect to women whose delivery was not attended by staff for treatment of
        # both causes of PPH
        def assessment_and_treatment_of_pph_retained_placenta():
            if outcome_of_request_for_consumables_pph:
                if params['prob_successful_manual_removal_placenta'] > self.module.rng.random_sample():
                    if mni[person_id]['delivery_attended']:
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                    else:
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                else:
                    if mni[person_id]['delivery_attended']:
                        mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                        mni[person_id]['referred_for_blood'] = 'prompt_referral'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                        mni[person_id]['referred_for_blood'] = 'delayed_referral'

        def assessment_and_treatment_of_pph_uterine_atony():
            # TODO: refine to 1 treatment effect (medical)?
            if outcome_of_request_for_consumables_pph:
                if mni[person_id]['delivery_attended']:
                    if params['prob_cure_oxytocin'] > self.module.rng.random_sample():
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                    elif params['prob_cure_misoprostol'] > self.module.rng.random_sample():
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                    elif params['prob_cure_uterine_massage'] > self.module.rng.random_sample():
                                df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                        mni[person_id]['referred_for_blood'] = 'prompt_referral'
                else:
                    if params['prob_cure_oxytocin'] > self.module.rng.random_sample():
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                    elif params['prob_cure_misoprostol'] > self.module.rng.random_sample():
                         df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                    elif params['prob_cure_uterine_massage'] > self.module.rng.random_sample():
                                df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                        mni[person_id]['referred_for_blood'] = 'delayed_referral'

            if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'retained_placenta':
                assessment_and_treatment_of_pph_retained_placenta()
            elif df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'uterine_atony':
                assessment_and_treatment_of_pph_uterine_atony()

    # ============================================== REFERRAL =========================================================
        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        # TODO: reset delivery_setting to home_birth and send back to postpartum event?
        pass


class HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1. This event is scheduled by the
    PostpartumLabourEvent . This event manages initial care around the time of delivery including prophylactic
    interventions (i.e. clean birth practices) for women presenting at Level 1 of the health system for delivery care.
     This event uses a womans stored risk of complications, which may be manipulated by treatment effects to determines
     if they will experience a complication during their labour in hospital. It is responsible for scheduling treatment
     HSIs for those complications.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'hospital'

        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1: Providing skilled attendance '
                    'following birth for person %d', person_id)

        # ================================= STORING COMPLICATION RISKS IN MNI =========================================
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='eclampsia',
                                                                      labour_stage='pp')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='postpartum_haem',
                                                                      labour_stage='pp')
            self.module.calculate_complication_risk_facility_delivery(person_id, complication='sepsis',
                                                                      labour_stage='pp')

        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            self.module.active_management_of_the_third_stage_of_labour()
        else:
            logger.debug('mother %d did not receive active management of the third stage of labour due to resource '
                         'constraints')

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        if ~mni[person_id]['sought_care_for_complication']:
            self.module.set_complications_during_facility_birth_postpartum(person_id, complication='eclampsia',
                                                                           labour_stage='pp')

            self.module.set_complications_during_facility_birth_postpartum(person_id, complication='postpartum_haem',
                                                                           labour_stage='pp')
            if ~df.at[person_id, 'la_sepsis']:
                self.module.set_complications_during_facility_birth_postpartum(person_id, complication='sepsis',
                                                                               labour_stage='pp')

    # ===================================== COMPLICATION ASSESSMENT ===========================================
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_maternal_sepsis('hp', 'pp')

        self.module.assessment_and_treatment_of_eclampsia()

        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'retained_placenta':
            self.module.assessment_and_treatment_of_pph_retained_placenta()
        elif df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'uterine_atony':
            self.module.assessment_and_treatment_of_pph_uterine_atony()

    # ============================================== REFERRAL =========================================================

        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusionFacilityLevel1(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint


class HSI_Labour_CaesareanSectionFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_Labour_CaesareanSectionFacilityLevel1. It is scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel1. This event manages caesarean section delivery.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_CaesareanSectionFacilityLevel1'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        logger.info('This is HSI_Labour_CaesareanSection: Person %d will now undergo delivery via Caesarean Section',
                    person_id)
        logger.info('%s|caesarean_section|%s', self.sim.date,
                    {'person_id': person_id})

        # We define the consumables needed for this event
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed_cs = {'Intervention_Package_Code': {pkg_code_cs: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_cs)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_cs]:
            logger.debug('All the required consumables are available and will can be used for this caesarean delivery.')
        else:
            logger.debug('The required consumables are not available for this delivery')

        # Set the variables to indicate this woman has undergone a caesarean delivery...
        mni[person_id]['mode_of_delivery'] = 'caesarean_section'
        df.at[person_id, 'la_previous_cs_delivery'] = True

        # TODO: effect of prophylactic antibiotics?
        # TODO: effect of being referred?

        # We set the treatment variables for those complications for which caesarean section is part of the treatment...
        if df.at[person_id, 'la_obstructed_labour'] and mni[person_id]['referred_for_cs'] == 'delayed_referral':
            df.at[person_id, 'la_obstructed_labour_treatment'] = 'delayed_treatment'
        elif df.at[person_id, 'la_obstructed_labour'] and mni[person_id]['referred_for_cs'] == 'prompt_referral':
            df.at[person_id, 'la_obstructed_labour_treatment'] = 'prompt_treatment'

        if df.at[person_id, 'la_antepartum_haem'] and mni[person_id]['referred_for_cs'] == 'delayed_referral':
            df.at[person_id, 'la_antepartum_haem_treatment'] = 'delayed_treatment'
        elif df.at[person_id, 'la_antepartum_haem'] and mni[person_id]['referred_for_cs'] == 'prompt_referral':
            df.at[person_id, 'la_antepartum_haem_treatment'] = 'prompt_treatment'

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'deliver via caesarean section', person_id, self.sim.date)

class HSI_Labour_CaesareanSectionFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_Labour_CaesareanSectionFacilityLevel2. It is scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel?. This event manages caesarean section delivery.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_CaesareanSectionFacilityLevel2'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        logger.info('This is HSI_Labour_CaesareanSection: Person %d will now undergo delivery via Caesarean Section',
                    person_id)
        logger.info('%s|caesarean_section|%s', self.sim.date,
                    {'person_id': person_id})

        # We define the consumables needed for this event
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed_cs = {'Intervention_Package_Code': {pkg_code_cs: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_cs)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_cs]:
            logger.debug('All the required consumables are available and will can be used for this caesarean delivery.')
        else:
            logger.debug('The required consumables are not available for this delivery')

        # Set the variables to indicate this woman has undergone a caesarean delivery...
        mni[person_id]['mode_of_delivery'] = 'caesarean_section'
        df.at[person_id, 'la_previous_cs_delivery'] = True

        # We set the treatment variables for those complications for which caesarean section is part of the treatment...
        if df.at[person_id, 'la_obstructed_labour'] and mni[person_id]['referred_for_cs'] == 'delayed_referral':
            df.at[person_id, 'la_obstructed_labour_treatment'] = 'delayed_treatment'
        elif df.at[person_id, 'la_obstructed_labour'] and mni[person_id]['referred_for_cs'] == 'prompt_referral':
            df.at[person_id, 'la_obstructed_labour_treatment'] = 'prompt_treatment'

        if df.at[person_id, 'la_antepartum_haem'] and mni[person_id]['referred_for_cs'] == 'delayed_referral':
            df.at[person_id, 'la_antepartum_haem_treatment'] = 'delayed_treatment'
        elif df.at[person_id, 'la_antepartum_haem'] and mni[person_id]['referred_for_cs'] == 'prompt_referral':
            df.at[person_id, 'la_antepartum_haem_treatment'] = 'prompt_treatment'

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'deliver via caesarean section', person_id, self.sim.date)


class HSI_Labour_ReceivesBloodTransfusionFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel1. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel1 or HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1
    This event manages blood transfusion for women who have experienced significant blood loss in labour"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesBloodTransfusionFacilityLevel1'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel1: Person %d will now receive a blood '
                    'transfusion following haemorrhage/blood loss delivery', person_id)
        # TODO: should this package be the minimum amount of blood needed? Also plts/ffp?
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_bt2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                  'Item_Code'])[0]
        consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1 ,
                                                                                item_code_bt3: 1, item_code_bt4: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_bt)

        # TODO: this needs to effect death
        # TODO: how should 'delayed referral' work here?

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


class HSI_Labour_ReceivesBloodTransfusionFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel2. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel? or HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1
    This event manages blood transfusion for women who have experienced significant blood loss in labour"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesBloodTransfusionFacilityLevel1'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel1: Person %d will now receive a blood '
                    'transfusion following haemorrhage/blood loss delivery', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_bt2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                  'Item_Code'])[0]
        consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1 ,
                                                                                item_code_bt3: 1, item_code_bt4: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_bt)

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


class HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_SurgeryForLabourComplications. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel1 or HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1
    This event manages surgery for women who have developed complication in labour where medical management has failed.
    This includes uterine rupture, antepartum haemorrhage, and postpartum haemorrhage."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_SurgeryForLabourComplicationsFacilityLevel1'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        logger.info('This is HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1: Person %d will now undergo surgery'
                    'for complications developed in labour', person_id)
        #TODO: Consumables...

        # Interventions...
        treatment_success_ur = params['success_rate_uterine_repair'] < self.module.rng.random_sample()

        if df.at[person_id, 'la_uterine_rupture']:
            if treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'prompt_treatment'
            elif treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'delayed_treatment'
            elif ~treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'prompt_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True
            elif ~treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'delayed_treatment'
                # todo: should this variable be turned on as well?
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        treatment_success_pph = params['success_rate_pph_surgery'] < self.module.rng.random_sample()

        if df.at[person_id, 'la_postpartum_haem']:
            if treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
            elif treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
            elif ~treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True
            elif ~treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True
        # todo: this will possibly give women a hysterectomy if products are retained, thats not right?
        # todo: failed firstline PPH treatment should be more at risk of death?
        # todo: should the successrate of removing a retained placenta be seperated?


    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'receive surgical care for her complication in labour', person_id, self.sim.date)


class HSI_Labour_SurgeryForLabourComplicationsFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_SurgeryForLabourComplications. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabourFacilityLevel1 or HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1
    This event manages surgery for women who have developed complication in labour where medical management has failed.
    This includes uterine rupture, antepartum haemorrhage, and postpartum haemorrhage."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_SurgeryForLabourComplicationsFacilityLevel2'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        logger.info('This is HSI_Labour_SurgeryForLabourComplicationsFacilityLevel1: Person %d will now undergo surgery'
                    'for complications developed in labour', person_id)

        # Interventions...
        treatment_success_ur = params['success_rate_uterine_repair'] < self.module.rng.random_sample()

        if df.at[person_id, 'la_uterine_rupture']:
            if treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'prompt_treatment'
            elif treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'delayed_treatment'
            elif ~treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'prompt_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True
            elif ~treatment_success_ur and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_uterine_rupture_treatment'] = 'delayed_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        treatment_success_pph = params['success_rate_pph_surgery'] < self.module.rng.random_sample()

        if df.at[person_id, 'la_postpartum_haem']:
            if treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
            elif treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
            elif ~treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'prompt_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True
            elif ~treatment_success_pph and mni[person_id]['referred_for_surgery'] == 'delayed_referral':
                df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                df.at[person_id, 'la_has_had_hysterectomy'] = True

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'receive surgical care for her complication in labour', person_id, self.sim.date)


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is LabourLoggingEvent. Currently it calculates and produces a yearly output of maternal mortality (maternal
    deaths per 100,000 live births). It is incomplete"""
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        df = self.sim.population.props

        # n.b. this is all cause/time-point maternal death (will need to be focused intrapartum)

        # Maternal Mortality Ratio
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births_sum = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        maternal_deaths = len(df.index[df.la_maternal_death_in_labour & (df.la_maternal_death_in_labour_date > one_year_prior) &
                                       (df.la_maternal_death_in_labour_date < self.sim.date)])

    #    if maternal_deaths == 0:
    #        mmr = 0
    #    else:
    #        mmr = maternal_deaths/live_births_sum * 100000

    #    logger.info(f'The maternal mortality for this year date %s is {mmr} per 100,000 live births', self.sim.date)

        # Facility Delivery Rate

        # Still Birth Rate
        # Perinatal Mortality
        # Disease Incidence
        # Intervention incidence
