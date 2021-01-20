from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.postnatal_supervisor import PostnatalWeekOneEvent
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_PresentsForCareWithSymptoms
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.USES_SYMPTOMMANAGER
    }

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
        'prob_pl_ol': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'prob_cephalopelvic_dis': Parameter(
            Types.REAL, 'an individuals probability of experiencing CPD'),
        'prob_malpresentation': Parameter(
            Types.REAL, 'an individuals probability of experiencing malpresentation'),
        'prob_malposition': Parameter(
            Types.REAL, 'an individuals probability of experiencing malposition'),
        'prob_obstruction_cpd': Parameter(
            Types.REAL, 'risk of obstruction in a woman with CPD'),
        'prob_obstruction_malpos': Parameter(
            Types.REAL, 'risk of obstruction in a woman with malposition'),
        'prob_obstruction_malpres': Parameter(
            Types.REAL, 'risk of obstruction in a woman with malpresentation'),
        'prob_placental_abruption_during_labour': Parameter(
            Types.REAL, 'probability of a woman developing placental abruption during labour'),
        'prob_aph_placenta_praevia_labour': Parameter(
            Types.REAL, 'probability of a woman with placenta praevia experiencing an APH during labour'),
        'prob_aph_placental_abruption_labour': Parameter(
            Types.REAL, 'probability of a woman with placental abruption experiencing an APH during labour'),
        'prob_chorioamnionitis_ip': Parameter(
            Types.REAL, 'probability of chorioamnionitis infection during labour'),
        'prob_other_maternal_infection_ip': Parameter(
            Types.REAL, 'probability of other obstetric infection in labour'),
        'prob_endometritis_pp': Parameter(
            Types.REAL, 'probability of endometritis infection following labour'),
        'prob_skin_soft_tissue_inf_pp': Parameter(
            Types.REAL, 'probability of a skin or soft tissue infection following labour'),
        'prob_urinary_tract_inf_pp': Parameter(
            Types.REAL, 'probability of a urinary tract infection following labour'),
        'prob_other_maternal_infection_pp': Parameter(
            Types.REAL, 'probability of other obstetric infections following labour'),
        'prob_sepsis_chorioamnionitis': Parameter(
            Types.REAL, 'risk of sepsis following chorioamnionitis infection'),
        'prob_sepsis_other_maternal_infection_ip': Parameter(
            Types.REAL, 'risk of sepsis following other intrapartum infection'),
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
        'prob_ip_still_birth_unk_cause': Parameter(
            Types.REAL, 'baseline probability of intrapartum still birth secondary to unknown cause'),
        'rr_still_birth_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth in mothers who have died during labour'),
        'rr_still_birth_aph': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing antepartum haemorrhage'),
        'rr_still_birth_ol': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing obstructed labour'),
        'rr_still_birth_ur': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing uterine rupture'),
        'rr_still_birth_sepsis': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing intrapartum sepsis'),
        'rr_still_birth_spe': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing severe pre-eclampsia'),
        'rr_still_birth_ec': Parameter(
            Types.REAL, 'relative risk of still birth in mothers experiencing eclampsia'),
        'prob_uterine_atony': Parameter(
            Types.REAL, 'probability of uterine atony following delivery'),
        'prob_lacerations': Parameter(
            Types.REAL, 'probability of genital tract lacerations following delivery'),
        'prob_retained_placenta': Parameter(
            Types.REAL, 'probability of placental retention following delievery'),
        'prob_other_pph_cause': Parameter(
            Types.REAL, 'probability of other pph causing factors'),
        'prob_pph_uterine_atony': Parameter(
            Types.REAL, 'risk of pph after experiencing uterine atony'),
        'prob_pph_lacerations': Parameter(
            Types.REAL, 'risk of pph after experiencing genital tract lacerations'),
        'prob_pph_retained_placenta': Parameter(
            Types.REAL, 'risk of pph after experiencing retained placenta'),
        'prob_pph_other_causes': Parameter(
            Types.REAL, 'risk of pph after experiencing otehr pph causes'),
        'prob_sepsis_endometritis': Parameter(
            Types.REAL, 'risk of sepsis following endometritis'),
        'prob_sepsis_urinary_tract_inf': Parameter(
            Types.REAL, 'risk of sepsis following urinary tract infection'),
        'prob_sepsis_skin_soft_tissue_inf': Parameter(
            Types.REAL, 'risk of sepsis following skin or soft tissue infection'),
        'prob_sepsis_other_maternal_infection_pp': Parameter(
            Types.REAL, 'risk of sepsis following other maternal postpartum infection'),
        'cfr_pp_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhage'),
        'rr_pph_death_anaemia': Parameter(
            Types.REAL, 'relative risk increase of death in women who are anaemic at time of PPH'),
        'cfr_pp_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pp_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis following delivery'),
        'prob_progression_severe_gest_htn': Parameter(
            Types.REAL, 'probability of severe gestational hypertension progressing to severe pre-eclampsia '
                        'during/after labour'),
        'prob_progression_mild_pre_eclamp': Parameter(
            Types.REAL, 'probability of mild pre-eclampsia progressing to severe pre-eclampsia during/after labour'),
        'prob_progression_severe_pre_eclamp': Parameter(
            Types.REAL, 'probability of severe pre-eclampsia progressing to eclampsia during/after labour'),


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
        'treatment_effect_maternal_infection_clean_delivery': Parameter(
            Types.REAL, 'Effect of clean delivery practices on risk of maternal infection'),
        'rr_pph_amtsl': Parameter(
            Types.REAL, 'relative risk of severe postpartum haemorrhage following active management of the third '
                        'stage of labour'),
        'prob_haemostatis_uterotonics': Parameter(
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
        'sensitivity_of_assessment_of_antepartum_haem_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment for referral by birth attendant for antepartum haemorrhage'
                        ' in a health centre'),
        'sensitivity_of_assessment_of_antepartum_haem_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment for treatment by birth attendant for antepartum haemorrhage'
                        ' in a level 1 hospital'),
        'sensitivity_of_assessment_of_uterine_rupture_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a health centre'),
        'sensitivity_of_assessment_of_uterine_rupture_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_ec_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for eclampsia in a level 1 hospital'),
        'sensitivity_of_assessment_of_ec_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_pph_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for postpartum haemorrhage in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_pph_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for postpartum haemorrhage in a level 1 '
                        'hospital'),
        'sepsis_treatment_effect_md': Parameter(
            Types.REAL, 'effect of treatment for sepsis on risk of maternal death'),
        'eclampsia_treatment_effect_severe_pe': Parameter(
            Types.REAL, 'effect of treatment for severe pre eclampsia on risk of eclampsia'),
        'eclampsia_treatment_effect_md': Parameter(
            Types.REAL, 'effect of treatment for eclampsia on risk of maternal death'),
        'anti_htns_treatment_effect_md': Parameter(
            Types.REAL, 'effect of IV anti hypertensive treatment on risk of death secondary to severe pre-eclampsia/'
                        'eclampsia stillbirth'),
        'aph_bt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of blood transfusion treatment for antepartum haemorrhage on risk of maternal death'),
        'pph_treatment_effect_uterotonics_md': Parameter(
            Types.REAL, 'effect of uterotonics on maternal death due to postpartum haemorrhage'),
        'pph_treatment_effect_mrp_md': Parameter(
            Types.REAL, 'effect of uterotonics on maternal death due to postpartum haemorrhage'),
        'pph_treatment_effect_surg_md': Parameter(
            Types.REAL, 'effect of surgery on maternal death due to postpartum haemorrhage'),
        'pph_treatment_effect_hyst_md': Parameter(
            Types.REAL, 'effect of hysterectomy on maternal death due to postpartum haemorrhage'),
        'pph_bt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of blood transfusion treatment for postpartum haemorrhage on risk of maternal death'),
        'aph_cs_treatment_effect_md': Parameter(
            Types.REAL, 'effect of caesarean section for antepartum haemorrhage on risk of maternal death'),
        'ur_repair_treatment_effect_md': Parameter(
            Types.REAL, 'effect of surgical uterine repair treatment for uterine rupture on risk of maternal death'),
        'ur_treatment_effect_bt_md': Parameter(
            Types.REAL, 'effect of blood transfusion treatment for uterine rupture on risk of maternal death'),
        'treatment_effect_avd_still_birth': Parameter(
            Types.REAL, 'effect of assisted vaginal delivery on risk of intrapartum still birth'),
        'treatment_effect_cs_still_birth': Parameter(
            Types.REAL, 'effect of caesarean section delivery on risk of intrapartum still birth'),
        'allowed_interventions': Parameter(
            Types.LIST, 'list of interventions allowed to run, used in analysis'),
        'squeeze_threshold_proph_ints': Parameter(
            Types.REAL, 'squeeze factor threshold below which prophylactic interventions for birth cant be given'),
        'squeeze_threshold_treatment_spe': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for severe pre-eclampsia cant be given'),
        'squeeze_threshold_treatment_ol': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for obstructed labour cant be given'),
        'squeeze_threshold_treatment_sep': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for maternal sepsis cant be given'),
        'squeeze_threshold_treatment_htn': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for hypertension cant be given'),
        'squeeze_threshold_treatment_ec': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for eclampsia cant be given'),
        'squeeze_threshold_treatment_ur': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for uterine rupture cant be given'),
        'squeeze_threshold_treatment_aph': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for antepartum haemorrhage cant be given'),
        'squeeze_threshold_treatment_pph': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for antepartum haemorrhage cant be given'),
        'squeeze_threshold_amtsl': Parameter(
            Types.REAL, 'squeeze factor threshold below which treatment for amtsl cant be given'),
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
        'la_obstructed_labour': Property(Types.BOOL, 'Whether this woman is experiencing obstructed labour'),
        'la_obstructed_labour_disab': Property(Types.BOOL, 'disability associated with obstructed labour'),
        'la_placental_abruption': Property(Types.BOOL, 'whether the woman has experienced placental abruption'),
        'la_antepartum_haem': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this'
                                                   'delivery'),
        'la_antepartum_haem_treatment': Property(Types.BOOL, 'whether this womans antepartum haemorrhage has been '
                                                             'treated'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this '
                                                   'delivery'),
        'la_uterine_rupture_disab': Property(Types.BOOL, 'disability associated with uterine rupture'),
        'la_uterine_rupture_treatment': Property(Types.BOOL, 'whether this womans uterine rupture has been treated'),
        'la_sepsis': Property(Types.BOOL, 'whether this woman has developed sepsis due to an intrapartum infection'),
        'la_sepsis_pp': Property(Types.BOOL, 'whether this woman has developed sepsis due to a postpartum infection'),
        'la_maternal_ip_infection': Property(Types.INT, 'bitset column holding list of infections'),
        'la_maternal_pp_infection': Property(Types.INT, 'bitset column holding list of postpartum infections'),
        # todo: this could be a list in MNI
        'la_sepsis_disab': Property(Types.BOOL, 'disability associated with maternal sepsis'),
        'la_sepsis_treatment': Property(Types.BOOL, 'If this woman has received treatment for maternal sepsis'),
        'la_eclampsia_disab': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        'la_eclampsia_treatment': Property(Types.BOOL, 'whether this womans eclampsia has been treated'),
        'la_severe_pre_eclampsia_treatment': Property(Types.BOOL, 'whether this woman has been treated for severe '
                                                                  'pre-eclampsia'),
        'la_maternal_hypertension_treatment': Property(Types.BOOL, 'whether this woman has been treated for maternal '
                                                                   'hypertension'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_cause': Property(Types.INT, 'bitset column holding causes of postpartum haemorrhage'),
        # todo: this could be a list in MNI
        'la_postpartum_haem_treatment': Property(Types.INT, ' Treatment for recevieved for postpartum haemorrhage '
                                                            '(bitset)'),
        'la_maternal_haem_non_severe_disab': Property(Types.BOOL, 'disability associated with non severe maternal '
                                                                  'haemorrhage'),
        'la_maternal_haem_severe_disab': Property(Types.BOOL, 'disability associated with severe maternal haemorrhage'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_maternal_death_in_labour': Property(Types.BOOL, ' whether the woman has died as a result of this '
                                                            'pregnancy'),
        'la_maternal_death_in_labour_date': Property(Types.DATE, 'date of death for a date in pregnancy'),
        'la_date_most_recent_delivery': Property(Types.DATE, 'date of on which this mother last delivered'),
        'la_is_postpartum': Property(Types.BOOL, 'Whether a woman is in the postpartum period, from delivery until '
                                                 'day +42 (6 weeks)'),
        'la_iron_folic_acid_postnatal': Property(Types.BOOL, 'Whether a woman is receiving iron and folic acid during '
                                                             'the postnatal period'),
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

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

            'obstructed_labour_ip': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('cpd', external=True).when(True, params['prob_obstruction_cpd']),
                Predictor('malposition', external=True).when(True, params['prob_obstruction_malpos']),
                Predictor('malpresentation', external=True).when(True, params['prob_obstruction_malpres'])),

            'chorioamnionitis_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_chorioamnionitis_ip'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'other_maternal_infection_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_other_maternal_infection_ip'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'endometritis_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_endometritis_pp'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'skin_soft_tissue_inf_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_skin_soft_tissue_inf_pp'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'urinary_tract_inf_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_urinary_tract_inf_pp'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'other_maternal_infection_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_other_maternal_infection_pp'],
                Predictor('received_clean_delivery', external=True).when(
                    True, params['treatment_effect_maternal_infection_clean_delivery'])),

            'sepsis_ip': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('ps_chorioamnionitis').when(True, params['prob_sepsis_chorioamnionitis']),
                Predictor('la_maternal_ip_infection').apply(
                    lambda x: params['prob_sepsis_chorioamnionitis']
                    if x & self.intrapartum_infections.element_repr('chorioamnionitis') else 0),
                Predictor('la_maternal_ip_infection').apply(
                    lambda x: params['prob_sepsis_other_maternal_infection_ip']
                    if x & self.intrapartum_infections.element_repr('other_maternal_infection') else 0)),

            'sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_sepsis'],
                Predictor('la_sepsis_treatment').when(True, params['sepsis_treatment_effect_md']),
                Predictor('ac_received_abx_for_chorioamnionitis').when(True, 0.5)),  # TODO: placeholder

            'sepsis_pp': LinearModel(
                LinearModelType.ADDITIVE,
                0,  # todo: ???
                Predictor('la_maternal_pp_infection').apply(
                    lambda x: params['prob_sepsis_endometritis']
                    if x & self.postpartum_infections.element_repr('endometritis') else 0),
                Predictor('la_maternal_pp_infection').apply(
                    lambda x: params['prob_sepsis_urinary_tract_inf']
                    if x & self.postpartum_infections.element_repr('urinary_tract_inf') else 0),
                Predictor('la_maternal_pp_infection').apply(
                    lambda x: params['prob_sepsis_skin_soft_tissue_inf']
                    if x & self.postpartum_infections.element_repr('skin_soft_tissue_inf') else 0),
                Predictor('la_maternal_pp_infection').apply(
                    lambda x: params['prob_sepsis_other_maternal_infection_pp']
                    if x & self.postpartum_infections.element_repr('other_maternal_infection') else 0)),

            'sepsis_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_sepsis'],
                Predictor('la_sepsis_treatment').when(True, params['sepsis_treatment_effect_md'])),

            'eclampsia_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_md']),
                Predictor('la_maternal_hypertension_treatment').when(True, params['anti_htns_treatment_effect_md']),
                Predictor('ac_iv_anti_htn_treatment').when(True, params['anti_htns_treatment_effect_md'])),

            'eclampsia_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_md'])),

            'severe_pre_eclamp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_severe_pre_eclamp'],
                Predictor('la_severe_pre_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_md']),
                Predictor('la_maternal_hypertension_treatment').when(True, params['anti_htns_treatment_effect_md']),
                Predictor('ac_iv_anti_htn_treatment').when(True, params['anti_htns_treatment_effect_md'])),

            'placental_abruption_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_placental_abruption_during_labour']),

            'antepartum_haem_ip': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('ps_placenta_praevia').when(True, params['prob_aph_placenta_praevia_labour']),
                # todo: multiplier to make this risk higher at point of delivery
                Predictor('ps_placental_abruption').when(True, params['prob_aph_placental_abruption_labour']),
                Predictor('la_placental_abruption').when(True, params['prob_aph_placental_abruption_labour'])),

            'antepartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_aph'],
                Predictor('received_blood_transfusion', external=True).when(True, params['aph_bt_treatment_effect_md']),
                Predictor('mode_of_delivery', external=True).when("caesarean_section",
                                                                  params['aph_cs_treatment_effect_md'])),

            'postpartum_haem_pp': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('la_postpartum_haem_cause').apply(
                    lambda x: params['prob_pph_uterine_atony']
                    if x & self.cause_of_primary_pph.element_repr('uterine_atony') else 0),
                Predictor('la_postpartum_haem_cause').apply(
                    lambda x: params['prob_pph_lacerations']
                    if x & self.cause_of_primary_pph.element_repr('lacerations') else 0),
                Predictor('la_postpartum_haem_cause').apply(
                    lambda x: params['prob_pph_retained_placenta']
                    if x & self.cause_of_primary_pph.element_repr('retained_placenta') else 0),
                Predictor('la_postpartum_haem_cause').apply(
                    lambda x: params['prob_pph_other_causes']
                    if x & self.cause_of_primary_pph.element_repr('other_pph_cause') else 0),
            ),

            'postpartum_haem_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_pph'],
                Predictor('ps_anaemia_in_pregnancy').when(True, params['rr_pph_death_anaemia']),
                Predictor('la_postpartum_haem_treatment').apply(
                    lambda x: params['pph_treatment_effect_uterotonics_md']
                    if x & self.pph_treatment.element_repr('uterotonics') else 1),
                Predictor('la_postpartum_haem_treatment').apply(
                    lambda x: params['pph_treatment_effect_mrp_md']
                    if x & self.pph_treatment.element_repr('manual_removal_placenta') else 1),
                Predictor('la_postpartum_haem_treatment').apply(
                    lambda x: params['pph_treatment_effect_surg_md']
                    if x & self.pph_treatment.element_repr('surgery') else 1),
                Predictor('la_postpartum_haem_treatment').apply(
                    lambda x: params['pph_treatment_effect_hyst_md']
                    if x & self.pph_treatment.element_repr('hysterectomy') else 1),
                Predictor('received_blood_transfusion', external=True).when(True, params['pph_bt_treatment_effect_md'])
            ),

            'uterine_rupture_ip': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_uterine_rupture']
                #   Predictor('la_parity').when('>4', params['or_ur_grand_multip']),
                #   Predictor('la_previous_cs_delivery').when(True, params['or_ur_prev_cs']),
                #   Predictor('la_obstructed_labour').when(True, params['or_ur_ref_ol']),
            ),

            'uterine_rupture_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_uterine_rupture'],
                Predictor('la_uterine_rupture_treatment').when(True, params['ur_repair_treatment_effect_md']),
                Predictor('received_blood_transfusion', external=True).when(True, params['ur_treatment_effect_bt_md'])),


            'intrapartum_still_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_still_birth_unk_cause'],
                Predictor('la_maternal_death_in_labour').when(True, params['rr_still_birth_maternal_death']),
                Predictor('la_antepartum_haem').when(True, params['rr_still_birth_aph']),
                Predictor('ps_antepartum_haemorrhage').when("mild_moderate", params['rr_still_birth_aph'])
                                                      .when("severe", params['rr_still_birth_aph']),
                Predictor('la_obstructed_labour').when(True,  params['rr_still_birth_ol']),
                Predictor('la_uterine_rupture').when(True,  params['rr_still_birth_ur']),
                Predictor('la_sepsis').when(True,  params['rr_still_birth_sepsis']),
                Predictor('ps_htn_disorders').when("severe_pre_eclamp",  params['rr_still_birth_spe'])
                                             .when("eclampsia",  params['rr_still_birth_ec']),
                Predictor('mode_of_delivery', external=True).when("instrumental",
                                                                  params['treatment_effect_avd_still_birth'])
                                                            .when("caesarean_section",
                                                                  params['treatment_effect_cs_still_birth'])),


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

            'probability_delivery_at_home': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_deliver_at_home'],
                Predictor('age_years').when('.between(34,40)', params['rrr_hb_delivery_age_35_39'])
                                      .when('.between(39,45)', params['rrr_hb_delivery_age_40_44'])
                                      .when('.between(44,50)', params['rrr_hb_delivery_age_45_49']),
                 Predictor('la_parity').when('.between(2,5)', params['rrr_hb_delivery_parity_3_to_4'])
                                      .when('>4', params['rrr_hb_delivery_parity_>4'])),


            'care_seeking_for_complication': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_careseeking_for_complication'],
                # Predictor('li_wealth').when('2', params['or_comp_careseeking_wealth_2']),
            ),
        }

            # todo: review 'cause' thinking for stillbirth to match the approach being used antenatally

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
        df.loc[df.is_alive, 'la_placental_abruption'] = False
        df.loc[df.is_alive, 'la_antepartum_haem'] = False
        df.loc[df.is_alive, 'la_antepartum_haem_treatment'] = False
        df.loc[df.is_alive, 'la_uterine_rupture'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_disab'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_treatment'] = False
        df.loc[df.is_alive, 'la_sepsis'] = False
        df.loc[df.is_alive, 'la_sepsis_pp'] = False
        df.loc[df.is_alive, 'la_sepsis_disab'] = False
        df.loc[df.is_alive, 'la_sepsis_treatment'] = False
        df.loc[df.is_alive, 'la_eclampsia_disab'] = False
        df.loc[df.is_alive, 'la_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_severe_pre_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_maternal_hypertension_treatment'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_maternal_haem_non_severe_disab'] = False
        df.loc[df.is_alive, 'la_maternal_haem_severe_disab'] = False
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour_date'] = pd.NaT
        df.loc[df.is_alive, 'la_date_most_recent_delivery'] = pd.NaT
        df.loc[df.is_alive, 'la_is_postpartum'] = False

        # Maternal sepsis and haemorrhage are outcomes of preceding clinical events. We use a bitset handler to manage
        # a property of the women that captures if she has experienced one or more of the necessary pre-ceding clinical
        # events that could trigger sepsis of haemorrhage

        self.intrapartum_infections = BitsetHandler(self.sim.population, 'la_maternal_ip_infection',
                                                    ['chorioamnionitis', 'other_maternal_infection'])

        self.postpartum_infections = BitsetHandler(self.sim.population, 'la_maternal_pp_infection',
                                                   ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                                                    'other_maternal_infection'])

        self.cause_of_primary_pph = BitsetHandler(self.sim.population, 'la_postpartum_haem_cause',
                                                  ['uterine_atony', 'lacerations', 'retained_placenta',
                                                   'other_pph_cause'])

        self.pph_treatment = BitsetHandler(self.sim.population, 'la_postpartum_haem_treatment',
                                           ['uterotonics', 'manual_removal_placenta', 'surgery', 'hysterectomy'])

        # TODO: it might be feasible to use the MNI dictionary to store these conditions as opposed to a property

        #  ----------------------------ASSIGNING PARITY AT BASELINE --------------------------------
        # We assign parity to all women of reproductive age at baseline
        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = np.around(
            params['la_labour_equations']['parity'].predict(df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)])
        )

    def initialise_simulation(self, sim):
        # We set the LoggingEvent to run a the last day of each year to produce statistics for that year
        sim.schedule_event(LabourLoggingEvent(self), sim.date + DateOffset(years=1))

        # This list contains all the women who are currently in labour and is used for checks/testing
        self.women_in_labour = []

        # This dictionary is the complication tracker used by the logger to output incidence of complications/outcomes
        self.labour_tracker = {'ip_stillbirth': 0, 'maternal_death': 0, 'obstructed_labour': 0,
                               'antepartum_haem': 0, 'antepartum_haem_death': 0, 'sepsis': 0, 'sepsis_death': 0,
                               'eclampsia': 0, 'severe_pre_eclampsia': 0, 'severe_pre_eclamp_death': 0,
                               'eclampsia_death': 0, 'uterine_rupture': 0,  'uterine_rupture_death': 0,
                               'postpartum_haem': 0, 'postpartum_haem_death': 0,
                               'sepsis_pp': 0, 'home_birth': 0, 'health_centre_birth': 0,
                               'hospital_birth': 0, 'caesarean_section': 0, 'early_preterm': 0,
                               'late_preterm': 0, 'post_term': 0, 'term': 0}

        # This list contains all possible complications/outcomes of the intrapartum and postpartum phase- its used in
        # assert functions as a test
        self.possible_intrapartum_complications = ['cephalopelvic_dis', 'malposition', 'malpresentation',
                                                   'obstructed_labour', 'placental_abruption',
                                                   'antepartum_haem', 'chorioamnionitis',
                                                   'other_maternal_infection', 'uterine_rupture', 'sepsis',
                                                   'eclampsia', 'severe_pre_eclamp']

        self.possible_postpartum_complications = ['sepsis', 'endometritis', 'skin_soft_tissue_inf', 'urinary_tract_inf',
                                                  'other_maternal_infection', 'sepsis_pp', 'uterine_atony',
                                                  'lacerations',
                                                  'retained_placenta', 'other_pph_cause', 'postpartum_haem',
                                                  'postpartum_haem_secondary', 'eclampsia', 'severe_pre_eclamp']

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
                property='la_sepsis_pp',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),

            assess_sepsis_hp_pp=DxTest(
                property='la_sepsis_pp',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),

            # Hypertension diagnosis
            assess_hypertension_hc=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hc']),

            assess_hypertension_hp=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hp']),

            # severe pre-eclampsia diagnosis
            assess_severe_pe_hc=DxTest(
                property='ps_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hc']),

            assess_severe_pe_hp=DxTest(
                property='ps_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hp']),

            # Eclampsia diagnosis
            assess_eclampsia_hc=DxTest(
                property='ps_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hc']),

            assess_eclampsia_hp=DxTest(
                property='ps_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hp']),

            # Antepartum Haemorrhage
            assess_aph_hc=DxTest(
                property='la_antepartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_antepartum_haem_hc']),

            assess_aph_hp=DxTest(
                property='la_antepartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_antepartum_haem_hc']),

            # Uterine Rupture
            assess_uterine_rupture_hc=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_assessment_of_uterine_rupture_hc']),

            assess_uterine_rupture_hp=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_assessment_of_uterine_rupture_hp']),

            # Postpartum haemorrhage
            assess_pph_hc=DxTest(
                property='la_postpartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_pph_hc']),

            assess_pph_hp=DxTest(
                property='la_postpartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_pph_hp']),
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
        df.at[child_id, 'la_placental_abruption'] = False
        df.at[child_id, 'la_antepartum_haem'] = False
        df.at[child_id, 'la_antepartum_haem_treatment'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_uterine_rupture_disab'] = False
        df.at[child_id, 'la_uterine_rupture_treatment'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_sepsis_pp'] = False
        df.at[child_id, 'la_sepsis_disab'] = False
        df.at[child_id, 'la_sepsis_treatment'] = False
        df.at[child_id, 'la_eclampsia_disab'] = False
        df.at[child_id, 'la_eclampsia_treatment'] = False
        df.at[child_id, 'la_severe_pre_eclampsia_treatment'] = False
        df.at[child_id, 'la_maternal_hypertension_treatment'] = False
        df.at[child_id, 'la_postpartum_haem'] = False
        df.at[child_id, 'la_maternal_haem_non_severe_disab'] = False
        df.at[child_id, 'la_maternal_haem_severe_disab'] = False
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_maternal_death_in_labour'] = False
        df.at[child_id, 'la_maternal_death_in_labour_date'] = pd.NaT
        df.at[child_id, 'la_date_most_recent_delivery'] = pd.NaT
        df.at[child_id, 'la_is_postpartum'] = False

        # If a mothers labour has resulted in an intrapartum still birth her child is still generated by the simulation
        # but the death is recorded through the InstantaneousDeath function

        # Store only live births to a mother parity
        if ~mother.la_intrapartum_still_birth:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity
        #    logger.info(key='live_birth', data={'mother': mother_id,
        #                                        'child': child_id})
        # TODO: review the logic of this thinking with Tim and Tim, very easy to map IP stillbirth without generating
        #  new row?
        if mother.la_intrapartum_still_birth:
            death = demography.InstantaneousDeath(self.sim.modules['Demography'],
                                                  child_id,
                                                  cause='intrapartum stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.at[mother_id, 'la_intrapartum_still_birth'] = False

        # We use this variable in the postnatal supervisor module to track postpartum women
        df.at[mother_id, 'la_is_postpartum'] = True
        df.at[mother_id, 'la_date_most_recent_delivery'] = self.sim.date

    def on_hsi_alert(self, person_id, treatment_id):
        """ This is called whenever there is an HSI event commissioned by one of the other disease modules."""
        logger.info(key='message', data=f'This is Labour, being alerted about a health system interaction '
                                        f'person {person_id }for: {treatment_id}')

    def report_daly_values(self):
        logger.debug(key='message', data='This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties data frame
        p = self.parameters

        # We women who experience complications have disability stored as a seperate property which is mapped to DALY
        # weights to capture disability.
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
        # TODO: (IMPORTANT) disability properties are being reset too early to be captured by this function -
        #  when reviewing disability at the model level discuss with TH how to proceed

    # ===================================== LABOUR SCHEDULER =======================================================

    def set_date_of_labour(self, individual_id):
        """This function, called by the contraception module, schedules a future date by which each woman who is
        pregnant will have gone into labour"""
        # TODO: This function should live in the pregnancy supervisor module- for neatness

        df = self.sim.population.props
        logger.debug(key='message', data=f'person {individual_id} is having their labour scheduled on date '
                                         f'{self.sim.date}',)

        # Check only alive newly pregnant women are scheduled to this function
        assert df.at[individual_id, 'is_alive'] and df.at[individual_id, 'is_pregnant']
        assert df.at[individual_id, 'date_of_last_pregnancy'] == self.sim.date

        # At the point of conception we schedule labour to onset for all women between 37 and 44 weeks gestation.
        # As a womans pregnancy progresses she has a risk of early labour onset applied from 24 weeks within the
        # pregnancy supervisor module
        df.at[individual_id, 'la_due_date_current_pregnancy'] = \
            (df.at[individual_id, 'date_of_last_pregnancy'] + pd.DateOffset(days=7 * 37 + self.rng.randint(0, 7 * 7)))

        self.sim.schedule_event(LabourOnsetEvent(self, individual_id),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])

        # Here we check that no one is scheduled to go into labour before 37 weeks gestation, ensuring all preterm
        # labour comes from the pregnancy supervisor module
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(259, unit='d')

    # ===================================== HELPER AND TESTING FUNCTIONS ==============================================

    def predict(self, eq, person_id):
        """This function compares the result of a specific linear equation with a random draw providing a boolean for
        the outcome under examination"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        person = df.loc[[person_id]]
        # TODO: Investigate if this can be replaced by the current LM predict function

        # We define specific external variables used as predictors in the equations defined below
        has_rbt = mni[person_id]['received_blood_transfusion']
        mode_of_delivery = mni[person_id]['mode_of_delivery']
        received_clean_delivery = mni[person_id]['clean_birth_practices']
        received_abx_for_prom = mni[person_id]['abx_for_prom_given']
        received_abx_for_pprom = mni[person_id]['abx_for_pprom_given']
        referral_timing_surgery = mni[person_id]['referred_for_surgery']
        referral_timing_caesarean = mni[person_id]['referred_for_cs']

        if 'cephalopelvic_dis' in mni[person_id]['obstructed_labour_cause']:
            cpd = True
        else:
            cpd = False

        if 'malposition' in mni[person_id]['obstructed_labour_cause']:
            malposition = True
        else:
            malposition = False

        if 'malpresentation' in mni[person_id]['obstructed_labour_cause']:
            malpresentation = True
        else:
            malpresentation = False

        return self.rng.random_sample() < eq.predict(person,
                                                     received_clean_delivery=received_clean_delivery,
                                                     received_abx_for_prom=received_abx_for_prom,
                                                     received_abx_for_pprom=received_abx_for_pprom,
                                                     mode_of_delivery=mode_of_delivery,
                                                     received_blood_transfusion=has_rbt,
                                                     referral_timing_surgery=referral_timing_surgery,
                                                     referral_timing_caesarean=referral_timing_caesarean,
                                                     cpd=cpd,
                                                     malpresentation=malpresentation,
                                                     malposition=malposition)[person_id]

    def reset_due_date(self, ind_or_df, id_or_index, new_due_date):
        df = self.sim.population.props

        if ind_or_df == 'individual':
            set = df.at
        else:
            set = df.loc

        set[id_or_index, 'la_due_date_current_pregnancy'] = new_due_date


    def check_labour_can_proceed(self, individual_id):
        df = self.sim.population.props
        person = df.loc[individual_id]

        if ~person.is_alive or ~person.is_pregnant or person.la_currently_in_labour:
            logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                             f'{self.sim.date}, however this is event is no longer relevant for this '
                                             f'individual and will not run')
            return False

        elif person.is_alive and person.is_pregnant and (person.la_due_date_current_pregnancy == self.sim.date) and \
             ~person.la_currently_in_labour:
            if person.ac_admitted_for_immediate_delivery == 'none':
                logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                                 f'{self.sim.date} and will now go into labour at gestation '
                                                 f'{person.ps_gestational_age_in_weeks}')
            else:
                logger.debug(key='message', data=f'person {individual_id}, who is currently admitted and awaiting '
                                                 f'delivery, has just gone into spontaneous labour and reached '
                                                 f'LabourOnsetEvent on {self.sim.date} - she will now go into labour at '
                                                 f'gestation {person.ps_gestational_age_in_weeks}')
            return True

        elif person.is_alive and person.is_pregnant and ~person.la_currently_in_labour and \
            (person.la_due_date_current_pregnancy != self.sim.date) and (person.ac_admitted_for_immediate_delivery !=
                                                                         'none'):

            logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                             f'{self.sim.date}- they have been admitted for delivery due to '
                                             f'complications in the antenatal period and will now progress into the '
                                             f'labour event at gestation {person.ps_gestational_age_in_weeks}')

            df.at[individual_id, 'la_due_date_current_pregnancy'] = self.sim.date
            return True

        else:
            return False

    def set_intrapartum_complications(self, individual_id, complication):
        """This function is called either during a home birth OR facility delivery to determine if a woman will
        experience the complication which has been passed to the function. In most cases this requires using the
        linear model and the predict function or simply a random draw against a set probability. The data frame is
        changed for the individual by this function, dependent on the result"""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info

        # TODO: RUN CHECK ON WOMEN ADMITTED FROM ANTENATAL PERIOD AND SET COMPLICATIONS ACCORDINGLY

        # First we run check to ensure only women who have started the labour process are passed to this function
        assert mni[individual_id]['delivery_setting'] != 'none'
        # Then we check that only complications from the master complication list are passed to the function (checks for
        # typos)
        assert complication in self.possible_intrapartum_complications

        if complication == 'antepartum_haem' and df.at[individual_id, 'ps_antepartum_haemorrhage'] != 'none':
            return

        if complication == 'placental_abruption' and df.at[individual_id, 'ps_placental_abruption']:
            return

        if complication == 'chorioamnionitis' and df.at[individual_id, 'ps_chorioamnionitis']:
            return

        # For the preceding complications that can cause obstructed labour, we apply risk as a set probability
        # TODO: replace with linear model equations following risk factor review?
        if complication == 'cephalopelvic_dis' or complication == 'malposition' or complication == 'malpresentation':
            result = self.rng.random_sample() < params[f'prob_{complication}']

        # Otherwise we use the linear model to predict likelihood of a complication
        else:
            result = self.predict(params['la_labour_equations'][f'{complication}_ip'], individual_id)

        # --------------------------------------- COMPLICATION ------------------------------------------------------
        if result:
            logger.debug(key='message', data=f'person {individual_id} has developed {complication} during birth on date'
                                             f'{self.sim.date}')

            # If 'result' == True, this woman will experience the complication passed to the function, and we make
            # changes to the data frame and store the complication in the tracker dictionary
            if complication == 'cephalopelvic_dis' or complication == 'malposition' or complication == 'mal' \
                                                                                                       'presentation':
                if 'none' in mni[individual_id]['obstructed_labour_cause']:
                    mni[individual_id]['obstructed_labour_cause'].remove('none')

                # Obstructed labour and sepsis have multiple potential preceding causes, presently these are stored as a
                # list and within a bitset    # TODO: Discuss best way to unify these (all lists/bitsets)
                mni[individual_id]['obstructed_labour_cause'].append(complication)

            elif complication == 'chorioamnionitis' or complication == 'other_maternal_infection':
                self.intrapartum_infections.set(individual_id, complication)

            elif complication == 'placental_abruption':
                if ~df.at[individual_id, 'ps_placental_abruption']:
                    df.at[individual_id, 'la_placental_abruption'] = True

            else:
                df.at[individual_id, f'la_{complication}'] = True
                self.labour_tracker[f'{complication}'] += 1

            # --------------------------------------- DISABILITY ----------------------------------------------------
            # Now we set disability properties for women with complications. Severity of bleeding is assigned if a
            # woman is experience an antepartum haemorrhage to map to DALY weights

            if complication == 'antepartum_haem':
                if df.at[individual_id, 'ps_antepartum_haemorrhage'] == 'none':
                    random_choice = self.rng.choice(['non_severe', 'severe'],
                                                    p=params['severity_maternal_haemorrhage'])
                    if random_choice == 'non_severe':
                        df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                    else:
                        df.at[individual_id, 'la_maternal_haem_severe_disab'] = True
                elif df.at[individual_id, 'ps_antepartum_haemorrhage'] == 'mild_moderate':
                    df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                elif df.at[individual_id, 'ps_antepartum_haemorrhage'] == 'severe':
                    df.at[individual_id, 'la_maternal_haem_severe_disab'] = True

        # todo: fix so only appropriate complications are triggering disab properties
        #    else:
        #        df.at[individual_id, f'la_{complication}_disab'] = True

    def set_postpartum_complications(self, individual_id, complication):
        """This function is called either following a home birth, facility delivery or caesarean section to determine
        if a woman will experience the complication which has been passed to the function. In most cases this requires
        using the linear model and the predict function or simply a random draw against a set probability.
        The data frame is changed for the individual by this function, dependent on the result"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        # This function follows a roughly similar pattern as set_intrapartum_complications
        assert mni[individual_id]['delivery_setting'] != 'none'
        assert complication in self.possible_postpartum_complications

        if complication == 'uterine_atony' or complication == 'retained_placenta':

            # Here we apply the effect of prophylactic treatment that may have been given following labour on risk of a
            # preceding cause of PPH
            if mni[individual_id]['amtsl_given']:

                # TODO: this might not be the correct place for this treatment effect
                risk_of_pph_cause = params[f'prob_{complication}'] * params['rr_pph_amtsl']
                result = risk_of_pph_cause < self.rng.random_sample()

            else:
                result = params[f'prob_{complication}'] < self.rng.random_sample()

        # Next we determine if this woman has experience lacerations or other potential causes of PPH
        elif complication == 'lacerations' or complication == 'other_pph_cause':
            result = self.rng.random_sample() < params[f'prob_{complication}']

        elif complication == 'sepsis_pp':
            # TODO: combine with rows below
            result = self.predict(params['la_labour_equations']['sepsis_pp'], individual_id)

        # Finally we use the linear model to calculate risk
        else:
            result = self.predict(params['la_labour_equations'][f'{complication}_pp'], individual_id)

        # ------------------------------------- COMPLICATION ---------------------------------------------------------
        if result:
            logger.debug(key='message', data=f'person {individual_id} has developed {complication} during the'
                                             f' postpartum phase of a birth on date {self.sim.date}')

            # If a woman will develop a postpartum infection we determine if this infection will onset immediately after
            # birth (< 2 days) or in the later postpartum period (2-42 days)
            if complication == 'endometritis' or complication == 'skin_soft_tissue_inf' or \
               complication == 'urinary_tract_inf' or complication == 'other_maternal_infection':
                self.postpartum_infections.set(individual_id, complication)

            if complication == 'sepsis_pp':

                # We run a check here to ensure only women with an infection can develop sepsis
                assert self.postpartum_infections.has_any([individual_id], 'endometritis', 'urinary_tract_inf',
                                                          'skin_soft_tissue_inf', 'other_maternal_infection',
                                                          first=True)

                df.at[individual_id, f'la_{complication}'] = True
                df.at[individual_id, 'la_sepsis_disab'] = True
                self.labour_tracker[f'{complication}'] += 1

            if complication == 'uterine_atony' or complication == 'lacerations' or complication == 'retained_placenta' \
               or complication == 'other_pph_cause':
                self.cause_of_primary_pph.set([individual_id], complication)

            if complication == 'postpartum_haem':

                # Similarly we make sure that PPH can only occur after a preceding cause
                assert self.cause_of_primary_pph.has_any([individual_id], 'uterine_atony', 'lacerations',
                                                         'retained_placenta', 'other_pph_cause', first=True)

                df.at[individual_id, f'la_{complication}'] = True
                self.labour_tracker[f'{complication}'] += 1

                random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                p=params['severity_maternal_haemorrhage'])

                if random_choice == 'non_severe':
                    df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[individual_id, 'la_maternal_haem_severe_disab'] = True

    def progression_of_hypertensive_disorders(self, individual_id):
        """This function is called during or following labour to determine if a woman with a hypertensive disorder will
         experience progression during this time period"""
        df = self.sim.population.props
        params = self.parameters

        # todo differentiation between ps_htn and pn_htn

        # TODO: property conventions
        # TODO: add in risk of progression from gest_htn to severe gest_htn, mitigated by treatment for hypertension

        # We only allow progression to the more severe states at this point as they will affect outcomes most
        if df.at[individual_id, 'ps_htn_disorders'] == 'severe_gest_htn':

            if params['prob_progression_severe_gest_htn'] > self.rng.random_sample():
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

        if df.at[individual_id, 'ps_htn_disorders'] == 'mild_pre_eclamp':

            if params['prob_progression_mild_pre_eclamp'] > self.rng.random_sample():
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
                self.labour_tracker['severe_pre_eclampsia'] += 1

        if df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':

            risk_ec = params['prob_progression_severe_pre_eclamp']

            if df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] or \
                (df.at[individual_id, 'ac_mag_sulph_treatment'] and
                 df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none'):
                risk_progression_spe_ec = risk_ec * params['eclampsia_treatment_effect_severe_pe']
            else:
                risk_progression_spe_ec = risk_ec

            if risk_progression_spe_ec > self.rng.random_sample():
                df.at[individual_id, 'ps_htn_disorders'] = 'eclampsia'
                df.at[individual_id, 'la_eclampsia_disab'] = True
                self.labour_tracker['eclampsia'] += 1

    def set_maternal_death_status_intrapartum(self, individual_id, cause):
        """This function calculates an associated risk of death for a woman who has experience a complication during
        labour and makes appropriate changes to the data frame"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        # TODO: could this method be replaced by a single additive linear model (how would treatment work)

        assert cause in self.possible_intrapartum_complications

        # First we determine if this woman will die of the complication defined in the function
        if self.predict(params['la_labour_equations'][f'{cause}_death'], individual_id):
            logger.debug(key='message', data=f'{cause} has contributed to person {individual_id} death during labour')

            mni[individual_id]['death_in_labour'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(cause)

            # This information is passed to the event where the InstantaneousDeathEvent is scheduled - this allows for
            # the application of multiple risk of death from multiple complications
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date
            self.labour_tracker[f'{cause}_death'] += 1

        else:
            # As eclampsia is a transient acute event, if women survive we reset their disease state to severe
            # pre-eclampsia
            if cause == 'eclampsia':
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

        #  And then a risk of stillbirth is calculated, maternal death is a used as a predictor and set very high if
        #  true
    #    if self.predict(params['la_labour_equations'][f'{cause}_stillbirth'], individual_id):
    #        logger.debug(key='message', data=f'person {individual_id} has experienced a still birth following {cause} '
    #                                         f'in labour')

    #        df.at[individual_id, 'la_intrapartum_still_birth'] = True
    #        df.at[individual_id, 'ps_prev_stillbirth'] = True
    #        df.at[individual_id, 'is_pregnant'] = False

    def set_maternal_death_status_postpartum(self, individual_id, cause):
        """This function calculates an associated risk of death for a woman who has experience a complication following
        labour and makes appropriate changes to the data frame"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        assert cause in self.possible_postpartum_complications

        if self.predict(params['la_labour_equations'][f'{cause}_pp_death'], individual_id):
            logger.debug(key='message',
                         data=f'{cause} has contributed to person {individual_id} death following labour')

            self.labour_tracker[f'{cause}_death'] += 1
            mni[individual_id]['death_postpartum'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date

        else:
            if cause == 'eclampsia':
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

    def apply_risk_of_early_postpartum_death(self, individual_id):
        """This function is called to apply risk of death to women who developed complications following delivery.
        It calls the set_maternal_death_status_postpartum for each complication and then schedules death accordinlgy"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info

        self.postpartum_characteristics_checker(individual_id)

        if df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
            self.set_maternal_death_status_postpartum(individual_id, cause='eclampsia')

        if df.at[individual_id, 'la_postpartum_haem']:
            self.set_maternal_death_status_postpartum(individual_id, cause='postpartum_haem')

        if df.at[individual_id, 'la_sepsis_pp']:
            self.set_maternal_death_status_postpartum(individual_id, cause='sepsis')

        if mni[individual_id]['death_postpartum']:
            self.labour_tracker['maternal_death'] += 1
            self.sim.schedule_event(demography.InstantaneousDeath(self, individual_id, cause='maternal'),
                                    self.sim.date)

            logger.debug(key='message', data=f'This is PostPartumDeathEvent scheduling a death for person '
                                             f'{individual_id} on date {self.sim.date} who died due to postpartum '
                                             f'complications')

        if mni[individual_id]['death_postpartum'] == False:
            # TODO: property convention
            df.at[individual_id, 'la_currently_in_labour'] = False
            self.intrapartum_infections.unset(
                [individual_id], 'chorioamnionitis', 'other_maternal_infection')
            self.postpartum_infections.unset(
                [individual_id], 'endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                'other_maternal_infection')
            df.at[individual_id, 'la_postpartum_haem'] = False
            df.at[individual_id, 'la_obstructed_labour'] = False
            df.at[individual_id, 'la_antepartum_haem'] = False
            df.at[individual_id, 'la_uterine_rupture'] = False

            df.at[individual_id, 'ps_placental_abruption'] = False
            df.at[individual_id, 'ps_placenta_praevia'] = False

            df.at[individual_id, 'la_sepsis_treatment'] = False
            df.at[individual_id, 'la_antepartum_haem_treatment'] = False
            df.at[individual_id, 'la_uterine_rupture_treatment'] = False
            df.at[individual_id, 'la_eclampsia_treatment'] = False
            df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] = False
            df.at[individual_id, 'la_maternal_hypertension_treatment'] = False

            # TODO: these should not be reset here as they wont be captured by DALY function
            df.at[individual_id, 'la_sepsis_disab'] = False
            df.at[individual_id, 'la_obstructed_labour_disab'] = False
            df.at[individual_id, 'la_uterine_rupture_disab'] = False
            df.at[individual_id, 'la_eclampsia_disab'] = False
            df.at[individual_id, 'la_maternal_haem_severe_disab'] = False
            df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = False

            # Reset this variable here so that in future pregnancies women still have risk applied through pregnancy
            # supervisor
            df.at[individual_id, 'ac_inpatient'] = False
            df.at[individual_id, 'ac_admitted_for_immediate_delivery'] = 'none' # todo: not the best place for this

            # todo: CALCULATE PERSONS DAYS POSTPARTUM AND USE THAT TO SCHEDULE BELOW (ENSURING AS EARLY AS
            #  POSSIBLE FOR ALL BUT >DAY1)

            # For women who have survived first 24 hours after birth we scheduled them to attend the first event in the
            # PostnatalSupervisorModule

            # This event determines if women/newborns will develop complications in week one. We stagger when women
            # arrive at this event to simulate bunching of complications in the first few days after birth

            days_post_birth_td = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']
            days_post_birth_int = int(days_post_birth_td / np.timedelta64(1, 'D'))

            assert days_post_birth_int < 6

            day_for_event = int(self.rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]))

            # Ensure no women go to this event after week 1
            if day_for_event + days_post_birth_int > 5:
                day_for_event = 0

            self.sim.schedule_event(PostnatalWeekOneEvent(self.sim.modules['PostnatalSupervisor'],
                                                                               individual_id),
                                    self.sim.date + DateOffset(days=day_for_event))

        # Here we remove all women (dead and alive) who have passed through the labour events
        self.women_in_labour.remove(individual_id)

    def labour_characteristics_checker(self, individual_id):
        """This function is called at different points in the module to ensure women of the right characteristics are
        in labour. This function doesnt check for a woman being pregnant or alive, as some events will still run despite
         those variables being set to false"""
        df = self.sim.population.props
        mother = df.loc[individual_id]

        print(individual_id)
        print(mother.ps_gestational_age_in_weeks)

        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour
        assert mother.ps_gestational_age_in_weeks >= 22

    def postpartum_characteristics_checker(self, individual_id):
        """This function is called at different points in the module to ensure women of the right characteristics are
        in labour. This function doesnt check for a woman being pregnant or alive, as some events will still run despite
         those variables being set to false"""
        df = self.sim.population.props
        mother = df.loc[individual_id]

        print(individual_id)
        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour
        assert mother.ps_gestational_age_in_weeks == 0
        assert mother.la_is_postpartum

    def events_queue_checker(self, individual_id):
        """ This function checks a womans event and hsi_event queues to ensure that she has the correct events scheduled
         during the process of labour"""
        mni = self.mother_and_newborn_info
        events = self.sim.find_events_for_person(person_id=individual_id)

        # TODO: remove prior to final PR, not currently used

        # Here we iterate through each womans event queue to insure she has the correct events scheduled
        # Firstly we check all women have the labour death event and birth event scheduled
        events = [e.__class__ for d, e in events]
        assert LabourDeathAndStillBirthEvent in events
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
                assert HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour in hsi_events

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

        # todo: house as separate functions

        if 'prophylactic_labour_interventions' not in params['allowed_interventions']:
            return
        else:
            # ----------------------------------CLEAN DELIVERY PRACTICES ---------------------------------------------
            pkg_code_clean_delivery_kit = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Clean practices and immediate essential newborn '
                                                                   'care (in facility)', 'Intervention_Pkg_Code'])[0]

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_clean_delivery_kit])

            if all_available:
                mni[person_id]['clean_birth_practices'] = True
                logger.debug(key='message', data=f'Mother {person_id} will be able to experience clean birth practices '
                                                 f'during her delivery as the consumables are available')
            else:
                logger.debug(key='message', data=f'Mother {person_id} will not be able to experience clean birth '
                                                 f'practices during her delivery as the consumables arent available')

            # --------------------------------- ANTIBIOTICS FOR PROM/PPROM -------------------------------------------
            if df.at[person_id, 'ps_premature_rupture_of_membranes']:

                if df.at[person_id, 'ac_received_abx_for_prom']:
                    mni[person_id]['abx_for_prom_given'] = True

                else:
                    pkg_code_pprom = pd.unique(
                        consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM',
                                        'Intervention_Pkg_Code'])[0]

                    all_available = hsi_event.get_all_consumables(
                        pkg_codes=[pkg_code_pprom])

                    if all_available:
                        mni[person_id]['abx_for_prom_given'] = True
                        logger.debug(key='message', data=f'This facility has provided antibiotics for mother '
                                                         f'{person_id} who is a risk of sepsis due to PROM')
                    else:
                        logger.debug(key='message', data='This facility has no antibiotics for the treatment of PROM.')

            # ------------------------------ STEROIDS FOR PRETERM LABOUR -------------------------------
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or mni[person_id]['labour_state'] == \
                'late_preterm_labour':

                item_code_steroids_prem_dexamethasone = pd.unique(
                    consumables.loc[consumables['Items'] == 'Dexamethasone 5mg/ml, 5ml_each_CMST', 'Item_Code'])[0]
                item_code_steroids_prem_betamethasone = pd.unique(
                    consumables.loc[consumables['Items'] == 'Betamethasone, 12 mg injection', 'Item_Code'])[0]

                consumables_steriod_preterm = {
                    'Intervention_Package_Code': {},
                    'Item_Code': {item_code_steroids_prem_dexamethasone: 1, item_code_steroids_prem_betamethasone: 1}}

                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_steriod_preterm)

                if (outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_dexamethasone]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_betamethasone]):
                    mni[person_id]['corticosteroids_given'] = True
                    logger.debug(key='message', data=f'This facility has provided corticosteroids for mother '
                                                     f'{person_id} who is in preterm labour')
                else:
                    logger.debug(key='message', data='This facility has no steroids for women in preterm labour.')

    def assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, hsi_event, facility_type):
        """This function defines the required consumables, determines correct diagnosis and administers an intervention
        to women suffering from severe pre-eclampsia in labour. It is called by
        HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        person_id = hsi_event.target

        if df.at[person_id, 'ac_admitted_for_immediate_delivery'] and df.at[person_id, 'ac_mag_sulph_treatment']:
            return

        if 'assessment_and_treatment_of_severe_pre_eclampsia' not in params['allowed_interventions']:
            return

        else:
            pkg_code_severe_preeclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                    'Intervention_Pkg_Code'])[0]
            all_available = hsi_event.get_all_consumables(
                    pkg_codes=[pkg_code_severe_preeclampsia])

            # Here we run a dx_test function to determine if the birth attendant will correctly identify this womans
            # severe pre-eclampsia, and therefore administer treatment
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run=f'assess_severe_pe_{facility_type}', hsi_event=hsi_event):

                if all_available:
                    df.at[person_id, 'la_severe_pre_eclampsia_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their severe pre-eclampsia '
                                                     f'identified during delivery. As consumables are available '
                                                     f'they will receive treatment')

                elif df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
                    logger.debug(key='message', data=f'mother {person_id} has not had their severe pre-eclampsia '
                                                     f'identified during delivery and will not be treated')

    def assessment_and_treatment_of_hypertension(self, hsi_event, facility_type):
        """This function defines the required consumables, determines correct diagnosis and administers intervention
        to women suffering from hypertension. It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if df.at[person_id, 'ac_iv_anti_htn_treatment']:
            return

        if 'assessment_and_treatment_of_hypertension' not in params['allowed_interventions']:
            return
        else:
            item_code_hydralazine = pd.unique(
                consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
                                'Item_Code'])[0]
            item_code_wfi = pd.unique(
                consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            item_code_gloves = pd.unique(
                consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

            consumables_gest_htn_treatment = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_hydralazine: 1,
                              item_code_wfi: 1,
                              item_code_gloves: 1,
                              item_code_needle: 1}}

            # Then query if these consumables are available during this HSI
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_gest_htn_treatment)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_hypertension_{facility_type}', hsi_event=hsi_event):

                # If they are available then the woman is started on treatment
                if (outcome_of_request_for_consumables['Item_Code'][item_code_hydralazine]) and \
                    (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
                    (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

                    df.at[person_id, 'la_maternal_hypertension_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their hypertension identified during '
                                                     f'delivery. As consumables are available they will receive'
                                                     f' treatment')

            elif df.at[person_id, 'ps_htn_disorders'] != 'none':
                logger.debug(key='message', data=f'mother {person_id} has not had their hypertension identified during '
                                                f'delivery and will not be treated')

    def assessment_and_treatment_of_eclampsia(self, hsi_event, facility_type):
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
            pkg_code_severe_preeclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]
            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_severe_preeclampsia])

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_eclampsia_{facility_type}', hsi_event=hsi_event):

                if all_available:
                    df.at[person_id, 'la_eclampsia_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their eclampsia identified during '
                                                     f'delivery. As consumables are available they will receive '
                                                     f'treatment')

                elif df.at[person_id, 'ps_htn_disorders'] == 'eclampsia':
                    logger.debug(key='message', data=f'mother {person_id} has not had their eclampsia identified '
                                                     f'during delivery and will not be treated')

    def assessment_and_treatment_of_obstructed_labour_via_avd(self, hsi_event, facility_type):
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

            outcome_of_request_for_consumables_ol = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_obstructed_labour)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_obstructed_labour_{facility_type}', hsi_event=hsi_event):

                if (outcome_of_request_for_consumables_ol['Intervention_Package_Code'][pkg_code_obstructed_labour]) and \
                    ((outcome_of_request_for_consumables_ol['Item_Code'][item_code_forceps]) or
                     (outcome_of_request_for_consumables_ol['Item_Code'][item_code_vacuum])):

                    logger.debug(key='message', data=f'mother {person_id} has had her obstructed labour identified'
                                                     f'during delivery. Staff will attempt an assisted vaginal delivery'
                                                     f'as the equipment is available')

                    # TODO: SHOULD TRUE ANY WOMAN WITH TRUE CPD IMMEDIATELY FAIL AVD and need CS
                    treatment_success = params['prob_successful_assisted_vaginal_delivery'] > self.rng.random_sample()

                    if treatment_success:
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    else:
                        # todo: is it too big an assumption that anyone for whome failed avd was attempted gets referred
                        #  for CS
                        logger.debug(key='message', data=f'Following a failed assisted vaginal delivery other '
                                                         f'{person_id} will need additional treatment')

                        mni[person_id]['referred_for_cs'] = True

            elif df.at[person_id, 'la_obstructed_labour']:
                logger.debug(key='message', data=f'mother {person_id} has not had their obstructed labour identified '
                                                 f'during delivery and will not be treated')

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

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_sepsis])

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_sepsis_{facility_type}_{labour_stage}', hsi_event=hsi_event):

                if all_available:
                    logger.debug(key='message', data=f'mother {person_id} has has their sepsis identified during '
                                                     f'delivery. As consumables are available they will receive '
                                                     f'treatment')

                    df.at[person_id, 'la_sepsis_treatment'] = True

            elif df.at[person_id, 'la_sepsis'] or df.at[person_id, 'la_sepsis_pp']:
                logger.debug(key='message', data=f'mother {person_id} has not had their sepsis identified during '
                                                 f'delivery and will not be treated')

    def assessment_and_plan_for_antepartum_haemorrhage(self, hsi_event, facility_type):
        """This function determines correct diagnosis and referral for intervention for women suffering from antepartum
        haemorrhage. It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_antepartum_haemorrhage' not in params['allowed_interventions']:
            return
        else:
            # todo: should we assume they will defs get treatment
            if df.at[person_id, 'ps_antepartum_haemorrhage'] != 'none' \
                and df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none':

                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['referred_for_blood'] = True
                logger.debug(key='message', data=f'mother {person_id} who was admitted for treatment following an '
                                                 f'antepartum haemorrhage will be referred for treatment ')
            else:
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                   dx_tests_to_run=f'assess_aph_{facility_type}',
                   hsi_event=hsi_event):
                    logger.debug(key='message', data=f'mother {person_id} has has their antepartum haemorrhage '
                                                     f'identified during delivery. They will now be referred for '
                                                     f'additional treatment')

                    mni[person_id]['referred_for_cs'] = True
                    mni[person_id]['referred_for_blood'] = True

                elif df.at[person_id, 'la_antepartum_haem']:
                    logger.debug(key='message', data=f'mother {person_id} has not had their antepartum haemorrhage '
                                                     f'identified during delivery and will not be referred for '
                                                     f'treatment')

    def assessment_for_referral_uterine_rupture(self, hsi_event, facility_type):
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
               dx_tests_to_run=f'assess_uterine_rupture_{facility_type}',
               hsi_event=hsi_event):
                logger.debug(key='message', data='mother %d has has their uterine rupture identified during delivery. '
                                                 'They will now be referred for additional treatment')

                mni[person_id]['referred_for_surgery'] = True
                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['referred_for_blood'] = True

            elif df.at[person_id, 'la_uterine_rupture']:
                logger.debug(key='message', data=f'mother {person_id} has not had their uterine_rupture identified '
                                                 f'during delivery and will not be referred for treatment')

    def active_management_of_the_third_stage_of_labour(self, hsi_event):
        """This function define consumables and administration of active management of the third stage of labour for
        women immediately following birth. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod"""
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'active_management_of_the_third_stage_of_labour' not in params['allowed_interventions']:
            return
        else:

            # todo: again hc/hp probability needed?
            pkg_code_am = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Active management of the 3rd stage of labour',
                                                                   'Intervention_Pkg_Code'])[0]

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_am])

            # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
            # labour (additional oxytocin, uterine massage and controlled cord traction)
            if all_available:
                logger.debug(key='message', data=f'mother {person_id} did not receive active management of the third '
                                                 f'stage of labour')
                mni[person_id]['amtsl_given'] = True
            else:
                logger.debug(key='message', data=f'mother {person_id} did not receive active management of the third '
                                                 f'stage of labour')

    def assessment_and_treatment_of_pph_uterine_atony(self, hsi_event, facility_type):
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

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_pph_{facility_type}', hsi_event=hsi_event):
                if outcome_of_request_for_consumables_pph:
                    if params['prob_haemostatis_uterotonics'] > self.rng.random_sample():
                        logger.debug(key='msg', data=f'mother {person_id} received uterotonics for her PPH which has '
                                                     f'resolved')
                        self.pph_treatment.set([person_id], 'uterotonics')
                        mni[person_id]['referred_for_blood'] = True

                    else:
                        logger.debug(key='msg', data=f'mother {person_id} received uterotonics for her PPH which has not'
                                                     f' resolved and she will need additional treatment')
                        mni[person_id]['referred_for_surgery'] = True
                        mni[person_id]['referred_for_blood'] = True

    def assessment_and_treatment_of_pph_retained_placenta(self, hsi_event, facility_type):
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

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_pph])

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_pph_{facility_type}', hsi_event=hsi_event):
                if all_available:
                    if params['prob_successful_manual_removal_placenta'] > self.rng.random_sample():
                        logger.debug(key='msg', data=f'mother {person_id} undergone MRP due to retained placenta and her'
                                                     f' PPH has resolved')
                        self.pph_treatment.set([person_id], 'manual_removal_placenta')
                        mni[person_id]['referred_for_blood'] = True

                    else:
                        logger.debug(key='msg',
                                     data=f'mother {person_id} undergone MRP due to retained placenta and her'
                                          f' PPH has not resolved- she will need futher treatment')
                        mni[person_id]['referred_for_surgery'] = True
                        mni[person_id]['referred_for_blood'] = True

    def interventions_delivered_pre_discharge(self, hsi_event):
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # todo: discuss with Tara (replace with in-event testing (not hsi))--
        if 'hiv' in self.sim.modules.keys():
            hiv_testing = HSI_Hiv_PresentsForCareWithSymptoms(
                module=self.sim.modules['hiv'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(hiv_testing, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # ------------------------------- Postnatal iron and folic acid ---------------------------------------------
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

        consumables_iron = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: 93}}  # days in 3 months

        # Check there availability
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_iron)

        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid]:
            df.at[person_id, 'la_iron_folic_acid_postnatal'] = True

        # TODO: link up with Tara and EPI module to code postnatal immunisation


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

        if self.module.check_labour_can_proceed(individual_id): # todo: add some kind of check for the very early
            # preterms admitted due to aph

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
                                  'delayed_pp_infection': False,
                                  'obstructed_labour_cause': ['none'],
                                  'onset_of_delayed_inf': 0,
                                  'corticosteroids_given': False,
                                  'clean_birth_practices': False,
                                  'abx_for_prom_given': False,
                                  'abx_for_pprom_given': False,
                                  'amtsl_given': False,
                                  'mode_of_delivery': 'vaginal_delivery',  # vaginal_delivery, instrumental,
                                  # caesarean_section
                                  'squeeze_to_high_for_hsi': False,
                                  'squeeze_to_high_for_hsi_pp': False,
                                  'sought_care_for_complication': False,
                                  'sought_care_labour_phase': 'none',  # none, intrapartum, postpartum
                                  'referred_for_cs': False,  # True (T) or False (F)
                                  'referred_for_blood': False,  # True (T) or False (F)
                                  'received_blood_transfusion': False,
                                  'referred_for_surgery': False,  # True (T) or False (F)'
                                  'death_in_labour': False,  # True (T) or False (F)
                                  'cause_of_death_in_labour': [],
                                  'stillbirth_in_labour': False,  # True (T) or False (F)
                                  'cause_of_stillbirth_in_labour': [],
                                  'death_postpartum': False,
                                  'dummy_counter': 0}  # True (T) or False (F)

            # ===================================== LABOUR STATE  ==================================
            # Next we calculate the number of days pregnant this woman is, and use that to log if her labour is early or
            # late preterm, term or post term

            foetal_age_in_days = (self.sim.date - df.at[individual_id, 'date_of_last_pregnancy']).days
            gestational_age_in_days = foetal_age_in_days + 14

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
            print(individual_id)
            print(foetal_age_in_days)
            print(gestational_age_in_days)
            print(df.at[individual_id, 'ps_gestational_age_in_weeks'])
            assert mni[individual_id]['labour_state'] is not None
            # todo: we're currently allowing 22 weekers to go into labour - this is wrong, we need to maybe say that if
            #  a woman is having a severe comp of pregnancy before 24 weeks that she has an induced abortion
            labour_state = mni[individual_id]['labour_state']
            logger.debug(key='message', data=f'This is LabourOnsetEvent, person {individual_id} has now gone into '
                                            f'{labour_state} on date {self.sim.date}')

            # ===================================== CARE SEEKING AND DELIVERY SETTING ==============
            # Only women who are in spontaneous labour, assumed to be in the community, can seek care.

            if df.at[individual_id, 'ac_admitted_for_immediate_delivery'] == 'none':

                # Here we calculate this womans predicted risk of home birth and health centre birth
                pred_hb_delivery = params['la_labour_equations']['probability_delivery_at_home'].predict(
                    df.loc[[individual_id]])[individual_id]
                pred_hc_delivery = params['la_labour_equations']['probability_delivery_health_centre'].predict(
                    df.loc[[individual_id]])[individual_id]

                # The denominator is calculated
                denom = 1 + pred_hb_delivery + pred_hc_delivery

                # Followed by the probability of each of the three outcomes - home birth, health centre birth or
                # hospital birth
                prob_hb = pred_hb_delivery / denom
                prob_hc = pred_hc_delivery / denom
                prob_hp = 1 / denom

                # And a probability weighted random draw is used to determine where the woman will deliver
                facility_types = ['home_birth', 'health_centre', 'hospital']
                probabilities = [prob_hb, prob_hc, prob_hp]
                mni[individual_id]['delivery_setting'] = self.module.rng.choice(facility_types, p=probabilities)

            else:
                # We assume all women with complications will deliver in a hospital
                mni[individual_id]['delivery_setting'] = 'hospital'

            # Check all women's 'delivery setting' is set
            assert mni[individual_id]['delivery_setting'] is not None

            # Women delivering at home move the the LabourAtHomeEvent as they will not receive skilled birth attendance
            if mni[individual_id]['delivery_setting'] == 'home_birth':
                self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)

                logger.info(key='message', data=f'This is LabourOnsetEvent, person {individual_id} as they has chosen '
                                                f'not to seek care at a health centre for delivery and will give birth '
                                                f'at home on date {self.sim.date}')

            # Otherwise the appropriate HSI is scheduled
            elif mni[individual_id]['delivery_setting'] == 'health_centre':
                health_centre_delivery = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=1)
                self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.info(key='message', data=f'This is LabourOnsetEvent, scheduling '
                                                f'HSI_Labour_PresentsForSkilledAttendanceInLabour on date '
                                                f'{self.sim.date} for person {individual_id} as they have chosen to '
                                                f'seek care at a health centre for delivery')

            elif mni[individual_id]['delivery_setting'] == 'hospital':
                # TODO: rng.choice is a placeholder for how to proceed with this aspect of care seeking
                facility_level = int(self.module.rng.choice([1, 2]))
                hospital_delivery = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=facility_level)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hospital_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))
                logger.info(key='message', data=f'This is LabourOnsetEvent, scheduling '
                                                f'HSI_Labour_PresentsForSkilledAttendanceInLabour on date '
                                                f'{self.sim.date} for person {individual_id} as they have chosen to '
                                                f'seek care at a hospital for delivery')

            # ======================================== SCHEDULING BIRTH AND DEATH EVENTS ===========
            # We schedule all women to move through the death event where those who have developed a complication
            # that hasn't been treated or treatment has failed will have a case fatality rate applied
            self.sim.schedule_event(LabourDeathAndStillBirthEvent(self.module, individual_id), self.sim.date +
                                    DateOffset(days=4))

            # Here we schedule the birth event for 4 days after labour- as women who die but still deliver a live child
            # will pass through birth event
            due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
            self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=5))
            logger.debug(key='message', data=f'This is LabourOnsetEvent scheduling a birth on date {due_date} to '
                                             f'mother {individual_id}')

            logger.debug(key='message', data=f'This is LabourOnsetEvent scheduling a potential death on date'
                                             f'{self.sim.date} for mother {individual_id}')

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
            logger.debug(key='message', data=f'person {individual_id} has is now going to deliver at home')
            self.module.labour_tracker['home_birth'] += 1

            # ===================================  APPLICATION OF COMPLICATIONS ====================
            # Using the complication_application function we loop through each complication and determine if a woman
            # will experience any of these if she has delivered at home

            for complication in ['cephalopelvic_dis', 'malpresentation', 'malposition', 'obstructed_labour',
                                 'placental_abruption', 'antepartum_haem', 'chorioamnionitis',
                                 'other_maternal_infection', 'sepsis', 'uterine_rupture']:
                self.module.set_intrapartum_complications(individual_id, complication=complication)
            self.module.progression_of_hypertensive_disorders(individual_id)

            # ==============================  CARE SEEKING FOLLOWING COMPLICATIONS =================
            # Here we determine if women who develop a complication during a home birth will seek care

            # (Women who have been scheduled a home birth after seeking care at a facility that didnt have capacity to
            # deliver the HSI will not try to seek care if they develop a complication)
            if not mni[individual_id]['squeeze_to_high_for_hsi']:
                if (individual.la_obstructed_labour or
                        individual.la_antepartum_haem or
                    (self.module.intrapartum_infections.has_any(
                        [individual_id], 'chorioamnionitis', 'other_maternal_infection', first=True)) or
                        individual.ps_htn_disorders == 'eclampsia' or
                        individual.ps_htn_disorders == 'severe_pre_eclamp' or
                        individual.la_uterine_rupture):

                    if self.module.predict(params['la_labour_equations']['care_seeking_for_complication'],
                                           individual_id):
                        mni[individual_id]['sought_care_for_complication'] = True
                        mni[individual_id]['sought_care_labour_phase'] = 'intrapartum'

                        # We assume women present to the health system through the emergency route
                        from tlo.methods.hsi_generic_first_appts import (
                            HSI_GenericEmergencyFirstApptAtFacilityLevel1)

                        event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                                module=self.module,
                                person_id=individual_id)

                        logger.debug(key='message', data=f'mother {individual_id} will now seek care for a complication'
                                                         f' that has developed during labour')
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))
                    else:
                        logger.debug(key='message', data=f'mother {individual_id} will not seek care following a '
                                                         f'complication that has developed during labour')


class BirthEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by LabourOnsetEvent. For women who survived labour, the appropriate
    variables are reset/updated and the function do_birth is executed. This event schedules PostPartumLabourEvent for
    those women who have survived"""

    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        df = self.sim.population.props
        person = df.loc[mother_id]
        mni = self.module.mother_and_newborn_info

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.info(key='message', data=f'mother {mother_id} at birth event')

        # Check the correct amount of time has passed between labour onset and birth event and that women at the event
        # have the right characteristics present
        assert (self.sim.date - df.at[mother_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.labour_characteristics_checker(mother_id)

        # =============================================== BIRTH ====================================================
        # If the mother is alive and still pregnant we generate a  child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications (intrapartum stillbirths till
        # trigger births for monitoring purposes)

        if person.is_alive and person.is_pregnant:
            logger.info(key='message', data=f'A Birth is now occurring, to mother {mother_id}')
            self.sim.do_birth(mother_id)

            self.sim.schedule_event(DeleteMNIDictionary(self.module, mother_id), self.sim.date + DateOffset(days=14))

        # ====================================== SCHEDULING POSTPARTUM EVENTS ======================================
            # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[mother_id]['delivery_setting'] == 'home_birth':
                self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, mother_id), self.sim.date)
                logger.debug(key='message', data=f'This is BirthEvent scheduling PostpartumLabourAtHomeEvent for '
                                                 f'person {mother_id} on date {self.sim.date}')

            # Women who deliver via caesarean follow a different event pathway so we differentiate here by delivery
            # type. Women post vaginal delivery attend the event below
            else:
                if mni[mother_id]['mode_of_delivery'] == 'vaginal_delivery' or \
                  mni[mother_id]['mode_of_delivery'] == 'instrumental':
                    health_centre_care = HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(
                        self.module, person_id=mother_id, facility_level_of_this_hsi=1)

                    all_facility_care = HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(
                        self.module, person_id=mother_id, facility_level_of_this_hsi=int(
                            self.module.rng.choice([1, 2])))

                    if mni[mother_id]['delivery_setting'] == 'health_centre':
                        logger.debug(key='message', data='This is BirthEvent scheduling HSI_Labour_ReceivesCareFor'
                                                         f'PostpartumPeriod for person {mother_id} on date'
                                                         f' {self.sim.date}')
                        self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_care,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))

                    elif mni[mother_id]['delivery_setting'] == 'hospital':
                        logger.info(key='message', data='This is BirthEvent scheduling '
                                                        'HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                                                        f'{mother_id}')

                        self.sim.modules['HealthSystem'].schedule_hsi_event(all_facility_care,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))
                else:
                    post_cs_care = HSI_Labour_ReceivesCareFollowingCaesareanSection(
                        self.module, person_id=mother_id)

                    logger.info(key='message', data='This is BirthEvent scheduling '
                                                    f'HSI_Labour_ReceivesCareFollowingCaesareanSection for person '
                                                    f'{mother_id}who gave birth via caesarean section  on date '
                                                    f'{self.sim.date}')
                    self.sim.modules['HealthSystem'].schedule_hsi_event(post_cs_care,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~person.is_alive and ~person.la_intrapartum_still_birth and person.la_maternal_death_in_labour:
            logger.info(key='message', data=f'A Birth is now occurring, to mother {mother_id} who died in childbirth '
                                            f'but her child survived')

            self.sim.do_birth(mother_id)
            # TODO: again for simplicity I think we could remove this


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
        # TODO: should I just use this method in all events (why is it used singularly here)
        if not df.at[individual_id, 'is_alive']:
            return

        # We first determine if this woman will experience any complications immediately following/ or in the days after
        # birth
        for complication in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf', 'other_maternal_infection',
                             'sepsis_pp', 'uterine_atony', 'lacerations', 'retained_placenta', 'other_pph_cause',
                             'postpartum_haem']:
            self.module.set_postpartum_complications(individual_id, complication=complication)

        self.module.progression_of_hypertensive_disorders(individual_id)

        # For women who experience a complication at home immediately following birth we use a care seeking equation to
        # determine if they will now seek additional care for management of this complication

        # Women who have come home, following a facility delivery, due to high squeeze will not try and seek care
        # for any complications
        if mni[individual_id]['squeeze_to_high_for_hsi_pp'] == False and \
            ((self.module.postpartum_infections.has_any(
                [individual_id], 'endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                'other_maternal_infection', first=True)) or df.at[individual_id, 'la_sepsis_pp']
                or df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia'
                or df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp'
                or df.at[individual_id, 'la_postpartum_haem']):

            # TODO: As the HTN SPE is not reset to mild PE after death event, women may seek care twice for SPE?

            if self.module.predict(params['la_labour_equations']['care_seeking_for_complication'], individual_id):

                # If this woman choses to seek care, she will present to the health system via the generic emergency
                # system and be referred on to receive specific care
                mni[individual_id]['sought_care_for_complication'] = True
                mni[individual_id]['sought_care_labour_phase'] = 'postpartum'

                from tlo.methods.hsi_generic_first_appts import (
                    HSI_GenericEmergencyFirstApptAtFacilityLevel1)

                event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                     module=self.module, person_id=individual_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    event,
                    priority=0,
                    topen=self.sim.date,
                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message', data=f'mother {individual_id} will now seek care for a complication that'
                                                 f' has developed following labour on date {self.sim.date}')
            else:
                logger.debug(key='message', data=f'mother {individual_id} will not seek care for a complication that'
                                                 f' has developed following labour on date {self.sim.date}')

                # For women who dont seek care for complications following birth we immediately apply risk of death
                self.module.apply_risk_of_early_postpartum_death(individual_id)


class LabourDeathAndStillBirthEvent (Event, IndividualScopeEventMixin):
    """This is the LabourDeathEvent. It is scheduled by the LabourOnsetEvent for all women who go through labour. This
    event determines if women who have experienced complications in labour will die or experience an intrapartum
    stillbirth."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # TODO: match causes of GBD nomenclature- see slack chat with TH

        # Check the correct amount of time has passed between labour onset and postpartum event
        print(individual_id)
        print(self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'])
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')
        self.module.labour_characteristics_checker(individual_id)

        if not df.at[individual_id, 'is_alive']:
            return

        # We cycle through each complication and apply risk of death using the following function
        if df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='severe_pre_eclamp')

        if df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='eclampsia')

        if df.at[individual_id, 'la_antepartum_haem'] or df.at[individual_id, 'ps_antepartum_haemorrhage']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='antepartum_haem')

        if df.at[individual_id, 'la_sepsis']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='sepsis')

        if df.at[individual_id, 'la_uterine_rupture']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='uterine_rupture')

        if self.module.predict(params['la_labour_equations']['intrapartum_still_birth'], individual_id):
            logger.debug(key='message', data=f'person {individual_id} has experienced an intrapartum still birth')

            df.at[individual_id, 'la_intrapartum_still_birth'] = True
            df.at[individual_id, 'ps_prev_stillbirth'] = True
            df.at[individual_id, 'is_pregnant'] = False
            # todo: reset antenatal variables

        # For a woman who dies of one or more of the above complications we schedule the death event
        if mni[individual_id]['death_in_labour']:
            self.module.labour_tracker['maternal_death'] += 1
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='maternal'), self.sim.date)

            # Log the maternal death
            logger.info(key='message', data=f'This is LabourDeathEvent scheduling a death for person {individual_id} on'
                                            f' date {self.sim.date} who died due to intrapartum complications')

            labour_complications = {'person_id': individual_id,
                                    'death_date': self.sim.date,
                                    'labour_profile': mni[individual_id]}

            logger.info(key='labour_complications', data=labour_complications, description='mni dictionary for a woman '
                                                                                           'who has died in labour')

            if mni[individual_id]['death_in_labour'] and df.at[individual_id, 'la_intrapartum_still_birth']:
                # We delete the mni dictionary if both mother and baby have died in labour, if the mother has died but
                # the baby has survived we delete the dictionary following the on_birth function of NewbornOutcomes
                del mni[individual_id]

        if df.at[individual_id, 'la_intrapartum_still_birth']:
            self.module.labour_tracker['ip_stillbirth'] += 1
            logger.info(key='message', data=f'A Still Birth has occurred, to mother {individual_id}')
            still_birth = {'mother_id': individual_id,
                           'date_of_ip_stillbirth': self.sim.date}

            logger.info(key='intrapartum_stillbirth', data=still_birth, description='record of intrapartum stillbirth')


class DeleteMNIDictionary (Event, IndividualScopeEventMixin):
    """Event scheduled for all women who deliver to ensure MNI dictionary is deleted- but still usable in modules that
    require information up until day + 14"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        if df.at[individual_id, 'is_alive']:
            del mni[individual_id]


class HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(HSI_Event, IndividualScopeEventMixin):
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
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        # TODO: DO WE ASSUME ALL WOMEN REFERRED FROM AN INPATIENT FOR DELIVERY GET DELIVERED

        if not df.at[person_id, 'is_alive']:
            return

        logger.info(key='message', data=f'This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Mother {person_id} '
                                        f'has presented to a health facility on date {self.sim.date} following the '
                                        f'onset of her labour')

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
        print(self.sim.date)
        print(df.at[person_id, 'la_due_date_current_pregnancy'])
        #assert self.sim.date == df.at[person_id, 'la_due_date_current_pregnancy'] or \
        #       self.sim.date == (df.at[person_id, 'la_due_date_current_pregnancy'] + pd.to_timedelta(1, unit='D'))

        if mni[person_id]['delivery_setting'] == 'health_centre':
            facility_type_code = 'hc'
        else:
            facility_type_code = 'hp'

        # LOG CONSUMABLES FOR DELIVERY
        pkg_code_delivery = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Vaginal delivery - skilled attendance',
                            'Intervention_Pkg_Code'])[0]

        self.get_all_consumables(
            pkg_codes=[pkg_code_delivery])

        # ===================================== PROPHYLACTIC CARE ===================================================
        # The following function manages the consumables and administration of prophylactic interventions in labour

        if squeeze_factor < params['squeeze_threshold_proph_ints']:
            self.module.prophylactic_labour_interventions(self)
        else:
            # Otherwise she receives no benefit of prophylaxis
            logger.debug(key='message', data=f'mother {person_id} did not receive prophylactic labour interventions due'
                                             f'to high squeeze')

        # ================================= PROPHYLACTIC MANAGEMENT PRE-ECLAMPSIA  ==============================

        if squeeze_factor < params['squeeze_threshold_treatment_spe']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, facility_type_code)
        else:
            logger.debug(key='message', data=f'mother {person_id} did not receive assessment or treatment of severe '
                                             f'pre-eclampsia due to high squeeze')

        # ===================================== APPLYING COMPLICATION INCIDENCE ====================
        # Following administration of prophylaxis (for attended deliveries) we assess if this woman will develop any
        # complications (effect of prophylaxis is included in the linear model for relevant complications)

        if not mni[person_id]['sought_care_for_complication']:
            if df.at[person_id, 'ps_chorioamnionitis']:
                self.module.intrapartum_infections.set([person_id], 'chorioamnionitis')

                for complication in ['cephalopelvic_dis', 'malpresentation', 'malposition',
                                     'obstructed_labour', 'placental_abruption', 'antepartum_haem',
                                     'other_maternal_infection', 'sepsis']:
                    self.module.set_intrapartum_complications(person_id, complication=complication)
            else:
                for complication in ['cephalopelvic_dis', 'malpresentation', 'malposition',
                                     'obstructed_labour', 'placental_abruption', 'antepartum_haem',
                                     'chorioamnionitis', 'other_maternal_infection', 'sepsis']:
                    self.module.set_intrapartum_complications(person_id, complication=complication)

            self.module.progression_of_hypertensive_disorders(person_id)

        # n.b. we do not apply the risk of uterine rupture due to the causal link between obstructed labour and
        # uterine rupture. We want interventions for obstructed labour to reduce the risk of uterine rupture

        # ======================================= COMPLICATION MANAGEMENT ==========================
        # Next, women in labour are assessed for complications and treatment delivered if a need is identified and
        # consumables are available

        if squeeze_factor < params['squeeze_threshold_treatment_ol']:
            self.module.assessment_and_treatment_of_obstructed_labour_via_avd(self, facility_type_code)

        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, facility_type_code, 'ip')

        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self, facility_type_code)

        if squeeze_factor < params['squeeze_threshold_treatment_aph']:
            self.module.assessment_and_plan_for_antepartum_haemorrhage(self, facility_type_code)

        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, facility_type_code)

        # Women in OL who arent treated are at risk of UR
        # todo: not sure about blocking UR like this
        if mni[person_id]['mode_of_delivery'] == 'vaginal_delivery' and mni[person_id]['referred_for_cs'] == False:
            self.module.set_intrapartum_complications(
                person_id, complication='uterine_rupture')

        # Uterine rupture follows the same pattern as antepartum haemorrhage
        if squeeze_factor < params['squeeze_threshold_treatment_ur']:
            self.module.assessment_for_referral_uterine_rupture(self, facility_type_code)

        # ============================================== REFERRAL ==================================
        # Finally we send any women who require additional treatment to the following HSIs

        if mni[person_id]['referred_for_cs'] or \
            mni[person_id]['referred_for_surgery'] or \
            mni[person_id]['referred_for_blood']:
            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL,
                timing='intrapartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # TODO: Women are only being sent to the same facility LEVEL (this doesnt reflect HC/DH referall to
        #  national hospitals)

        # If a this woman has experienced a complication the appointment footprint is changed from normal to
        # complicated
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if df.at[person_id, 'la_sepsis'] or df.at[person_id, 'la_antepartum_haem'] or \
            df.at[person_id, 'la_obstructed_labour'] or df.at[person_id, 'la_uterine_rupture']\
            or df.at[person_id, 'ps_htn_disorders'] == 'eclampsia' \
            or df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':

            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery']  # todo: is this right?

        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info

        # If a woman has chosen to deliver in a facility from the onset of labour, but the squeeze factor is too high,
        # she will be forced to return home to deliver
        if not mni[person_id]['sought_care_for_complication']:
            logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} '
                                             f'on date {self.sim.date} and she will now deliver at home')

            mni[person_id]['delivery_setting'] = 'home_birth'
            mni[person_id]['squeeze_to_high_for_hsi'] = True
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)

        # If a woman has presented to this event during labour due to a complication, she will not receive any treatment
        if mni[person_id]['sought_care_for_complication']:
            logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} '
                                             f'on date {self.sim.date} and she could not receive care for the '
                                             f'complications developed during her homebirth')

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
        logger.debug(key='message', data=f'This event is not in the allowed service availability and therefore cannot '
                                         f'run for mother {person_id} on date {self.sim.date}, she will now deliver at '
                                         f'home')

        if not mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'home_birth'
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)


class HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(HSI_Event, IndividualScopeEventMixin):
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
        params = self.module.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.info(key='message', data='This is HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1: Providing '
                                        f'skilled attendance following birth for person {person_id}')

        if not df.at[person_id, 'is_alive']:
            return

        assert mni[person_id]['mode_of_delivery'] == 'vaginal_delivery' or \
               mni[person_id]['mode_of_delivery'] == 'instrumental'

        # Although we change the delivery setting variable to 'facility_delivery' we do not include women who present
        # for care following birth, due to complications, as facility deliveries
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # We run similar checks as the labour HSI
        self.module.postpartum_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        # -------------------------- Active Management of the third stage of labour ----------------------------------
        # Prophylactic treatment to prevent postpartum bleeding is applied
        if mni[person_id]['sought_care_for_complication'] == False and squeeze_factor < \
            params['squeeze_threshold_amtsl']:
            self.module.active_management_of_the_third_stage_of_labour(self)

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Again we use the mothers individual risk of each complication to determine if she will experience any
        # complications using the set_complications_during_facility_birth function.
        if not mni[person_id]['sought_care_for_complication']:

            for complication in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                                 'other_maternal_infection', 'sepsis_pp',
                                 'uterine_atony', 'lacerations', 'retained_placenta', 'other_pph_cause',
                                 'postpartum_haem']:
                self.module.set_postpartum_complications(person_id, complication=complication)
            self.module.progression_of_hypertensive_disorders(person_id)

        # ======================================= COMPLICATION MANAGEMENT =============================================
        if mni[person_id]['delivery_setting'] == 'health_centre':
            facility_type_code = 'hc'
        else:
            facility_type_code = 'hp'

        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, facility_type_code, 'pp')
        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self, facility_type_code)
        if squeeze_factor < params['squeeze_threshold_treatment_pph']:
            self.module.assessment_and_treatment_of_pph_retained_placenta(self, facility_type_code)
            self.module.assessment_and_treatment_of_pph_uterine_atony(self, facility_type_code)
        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, facility_type_code)
        # TODO: treatment for lacerations?

        self.module.interventions_delivered_pre_discharge(self)

        # ============================================== REFERRAL ===================================================
        if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_blood']:
            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL,
                timing='postpartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # ====================================== APPLY RISK OF DEATH===================================================
        # For women who experience a complication following birth in a facility, but do not require additional care,
        # we apply risk of death considering the treatment effect applied (women referred on have this risk applied
        # later)
        if mni[person_id]['referred_for_surgery'] == False and mni[person_id]['referred_for_blood'] == False:

            if df.at[person_id, 'la_sepsis_pp'] or df.at[person_id, 'la_postpartum_haem'] or \
                df.at[person_id,'ps_htn_disorders'] == 'eclampsia':

                logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
                print(self)
                self.module.apply_risk_of_early_postpartum_death(person_id)


        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info

        logger.debug(key='message', data='HSI_Labour_ReceivesCareForPostpartumPeriod: did not run as the squeeze factor'
                                         f'is too high, mother {person_id} will return home on date {self.sim.date}')

        # TODO: i think if they have received care in labour we should assume they get this (i.e. remove did_not_run
        #  option)

        # Women who delivered at a facility, but can receive no more care due to high squeeze, will go home for the
        # immediate period after birth- where there risk of complications is applied
        if mni[person_id]['delivery_setting'] != 'home_birth':
            mni[person_id]['squeeze_to_high_for_hsi_pp'] = True
            self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, person_id), self.sim.date)

        return False

    def not_available(self):
        person_id = self.target
        logger.debug(key='message', data='This event is not in the allowed service availability and therefore cannot '
                                         f'run for mother {person_id} on date {self.sim.date}, she will now deliver at '
                                         f'home')


class HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(HSI_Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, person_id, facility_level_of_this_hsi, timing):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_SurgeryForLabourComplicationsFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

        self.timing = timing

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if mni[person_id]['referred_for_cs'] and self.timing == 'intrapartum':
            self.module.labour_tracker['caesarean_section'] += 1

            pkg_code_cs = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Cesearian Section with indication (with complication)',
                    'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[pkg_code_cs])

            if all_available:
                logger.debug(key='message',
                             data='All the required consumables are available and will can be used for this'
                                  'caesarean delivery.')
            else:
                logger.debug(key='message', data='The required consumables are not available for this delivery')

            mni[person_id]['mode_of_delivery'] = 'caesarean_section'
            mni[person_id]['amtsl_given'] = True
            df.at[person_id, 'la_previous_cs_delivery'] = True

            # -------------------------------------- RISK OF PPH POST CAESAREAN ---------------------------------------
            # TODO: currently assuming PPH following CS is due to atonic uterus or other (placenta is delivered)
            for complication in ['uterine_atony', 'other_pph_cause', 'postpartum_haem']:
                self.module.set_postpartum_complications(person_id, complication=complication)

            # -------------------------------------- TREATMENT OF PPH -------------------------------------------------

            self.module.assessment_and_treatment_of_pph_uterine_atony(self, 'hp')
            mni[person_id]['referred_for_blood'] = True

                # todo: this may cause mni[person_id]['referred_for_surgery'] to be true (does that make sense in the
                #  flow of the event)

        if mni[person_id]['referred_for_surgery'] and self.timing == 'intrapartum':

            dummy_surg_pkg_code = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                            'Cesearian Section with indication (with complication)',
                                                            'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[dummy_surg_pkg_code])

            if all_available:
                logger.debug(key='message',
                             data='Consumables required for surgery are available and therefore have been '
                                  'used')
            else:
                logger.debug(key='message',
                             data='Consumables required for surgery are unavailable and therefore have not '
                                  'been used')

            # Interventions...

            # Uterine Repair...
            if df.at[person_id, 'la_uterine_rupture']:
                treatment_success_ur = params['success_rate_uterine_repair'] < self.module.rng.random_sample()

                if treatment_success_ur:
                    df.at[person_id, 'la_uterine_rupture_treatment'] = True
                elif ~treatment_success_ur:
                    df.at[person_id, 'la_uterine_rupture_treatment'] = True
                    df.at[person_id, 'la_has_had_hysterectomy'] = True

        if mni[person_id]['referred_for_surgery'] and self.timing == 'postpartum':
                # Surgery for refractory atonic uterus...
                if df.at[person_id, 'la_postpartum_haem'] and self.module.cause_of_primary_pph.has_any([person_id],
                                                                                                       'uterine_atony',
                                                                                                       first=True):
                    treatment_success_pph = params['success_rate_pph_surgery'] < self.module.rng.random_sample()

                    if treatment_success_pph:
                        logger.debug(key='msg',
                                     data=f'mother {person_id} undergone surgery to manage her PPH which resolved')
                        self.module.pph_treatment.set(person_id, 'surgery')

                    elif ~treatment_success_pph:
                        logger.debug(key='msg',
                                     data=f'mother {person_id} undergone surgery to manage her PPH, she required a '
                                          f'hysterectomy to stop the bleeding')
                        self.module.pph_treatment.set(person_id, 'hysterectomy')
                        df.at[person_id, 'la_has_had_hysterectomy'] = True

                if df.at[person_id, 'la_postpartum_haem'] and self.module.cause_of_primary_pph.has_any(
                    [person_id], 'retained_placenta', first=True):
                    self.module.pph_treatment.set(person_id, 'surgery')
                    logger.debug(key='msg',
                             data=f'mother {person_id} undergone surgical removal of a retained placenta ')

        if mni[person_id]['referred_for_blood']:
            # todo: could this lead to repeat administration

            item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
            item_code_bt2 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
            item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
            item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                      'Item_Code'])[0]

            consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1,
                                                                                    item_code_bt3: 1, item_code_bt4: 2}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_bt)

            # If they're available, the event happens
            if (outcome_of_request_for_consumables['Item_Code'][item_code_bt1]) \
                and (outcome_of_request_for_consumables['Item_Code'][item_code_bt2]) \
                and (outcome_of_request_for_consumables['Item_Code'][item_code_bt3]) \
                and (outcome_of_request_for_consumables['Item_Code'][item_code_bt4]):

                mni[person_id]['received_blood_transfusion'] = True
                logger.debug(key='message', data=f'Mother {person_id} has received a blood transfusion due following a'
                                                 f' maternal haemorrhage')
            else:
                logger.debug(key='message', data=f'Mother {person_id} was unable to receive a blood transfusion due to '
                                                 f'insufficient consumables')

        if self.timing == 'postpartum':
            logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
            print(self)
            print(mni[person_id])
            self.module.apply_risk_of_early_postpartum_death(person_id)


        # todo: is this right
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

    #     if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_cs']:
    #        actual_appt_footprint['MajorSurg'] = actual_appt_footprint['MajorSurg']
    #    elif (~mni[person_id]['referred_for_surgery'] and ~mni[person_id]['referred_for_cs']) and \
    #        mni[person_id]['referred_for_blood']:
    #        actual_appt_footprint['MajorSurg'] = actual_appt_footprint['InpatientDays']


    def did_not_run(self):
        person_id = self.target
        logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} on '
                                         f'date {self.sim.date} and she is unable to deliver via caesarean section')

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
        logger.debug(key='message', data=f'this event is not allowed in the current service availability, for mother '
                                         f'{person_id} on date {self.sim.date} and she is unable to deliver via '
                                         f'caesarean section')

        self.module.set_intrapartum_complications(
            person_id, complication='uterine_rupture')
        return False


class HSI_Labour_ReceivesCareFollowingCaesareanSection(HSI_Event, IndividualScopeEventMixin):
    """."""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesCareFollowingCaesareanSection'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})  # todo: change
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        logger.debug(key='message', data='Labour_ReceivesCareFollowingCaesareanSection is firing')

        if not df.at[person_id, 'is_alive']:
            return

        assert mni[person_id]['mode_of_delivery'] == 'caesarean_section'
        assert mni[person_id]['referred_for_cs']
        # assert mni[person_id]['delivery_setting'] == 'hospital'  # TODO: set women to hospital who have CS?

        # This event represents care women receive after delivering via caesarean section
        # Women pass through different 'post delivery' events depending on mode of delivery due to how risk and
        # treatment of certain complications, such as post-partum haemorrhage, are managed

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Here we apply the risk that this woman will develop and infection or experience worsening hypertension after
        # her caesarean
        for complication in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                             'other_maternal_infection', 'sepsis_pp']:
            self.module.set_postpartum_complications(person_id, complication=complication)

        self.module.progression_of_hypertensive_disorders(person_id)

        # ======================================= COMPLICATION MANAGEMENT =============================================
        # Next we apply treatment effects
        self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hp', 'pp')
        if df.at[person_id, 'ps_htn_disorders'] == 'eclampsia':
            self.module.assessment_and_treatment_of_eclampsia(self)

        # ====================================== APPLY RISK OF DEATH===================================================
        # If this woman has any complications we apply risk of death accordingly

        if df.at[person_id, 'la_sepsis_pp'] or df.at[person_id, 'la_postpartum_haem']\
            or df.at[person_id, 'ps_htn_disorders'] == 'eclampsia':
            print(self)
            logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
            self.module.apply_risk_of_early_postpartum_death(person_id)


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
        spe = self.module.labour_tracker['severe_pre_eclampsia']
        pph = self.module.labour_tracker['postpartum_haem']
        sep = self.module.labour_tracker['sepsis']
        sep_pp = self.module.labour_tracker['sepsis_pp']

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
                           'ec_incidence': ec / total_births_last_year * 100,
                            'spe_incidence': spe/total_births_last_year * 100,
                           'sep_incidence': sep + sep_pp / total_births_last_year * 100,
                           'sep_incidence_pp': sep_pp / total_births_last_year * 100,
                           'pph_incidence': pph / total_births_last_year * 100,
                           }

        dict_crude_cases = {'intrapartum_mmr': total_ip_maternal_deaths_last_year,
                            'ol_cases': ol,
                            'aph_cases': aph,
                            'ur_cases': ur,
                            'ec_cases': ec,
                            'spe_cases': spe,
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

        logger.info(key='labour_summary_stats_incidence', data=dict_for_output, description='Yearly incidence summary '
                                                                                            'statistics output from '
                                                                                            'the labour module')
        logger.info(key='labour_summary_stats_crude_cases', data=dict_crude_cases, description='Yearly crude '
                                                                                               'summary statistics '
                                                                                               'output from the labour '
                                                                                               'module')
        logger.info(key='labour_summary_stats_delivery', data=deliveries, description='Yearly delivery summary '
                                                                                      'statistics output from the '
                                                                                      'labour module')
        logger.info(key='labour_summary_stats_death', data=deaths, description='Yearly death summary statistics '
                                                                               'output from the labour module')

        # Reset the EventTracker
        self.module.labour_tracker = {'ip_stillbirth': 0, 'maternal_death': 0, 'obstructed_labour': 0,
                                      'antepartum_haem': 0, 'antepartum_haem_death': 0, 'sepsis': 0, 'sepsis_death': 0,
                                      'eclampsia': 0, 'severe_pre_eclampsia':0, 'severe_pre_eclamp_death': 0,
                                      'eclampsia_death': 0, 'uterine_rupture': 0,  'uterine_rupture_death': 0,
                                      'postpartum_haem': 0, 'postpartum_haem_death': 0,
                                      'sepsis_pp': 0, 'home_birth': 0, 'health_centre_birth': 0,
                                      'hospital_birth': 0, 'caesarean_section': 0, 'early_preterm': 0,
                                      'late_preterm': 0, 'post_term': 0, 'term': 0}
