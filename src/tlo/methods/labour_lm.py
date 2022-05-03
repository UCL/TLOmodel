"""Module contains functions to be passed to LinearModel.custom function

The following template can be used for implementing:

def predict_for_individual(self, df, rng=None, **externals):
    # this is a single row dataframe. get the individual record.
    person = df.iloc[0]
    params = self.parameters
    result = 0.0  # or other intercept value
    # ...implement model here, adjusting result...
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)

or

def predict_for_dataframe(self, df, rng=None, **externals):
    params = self.parameters
    result = pd.Series(data=params['some_intercept'], index=df.index)
    # result series has same index as dataframe, update as required
    # e.g. result[df.age == 5.0] += params['some_value']
    return result
"""
import pandas as pd

from tlo.methods import pregnancy_helper_functions


def predict_parity(self, df, rng=None, **externals):
    """
    Population level linear model (additive) which returns a df containing the predicted parity (previous number of
    births) for all women aged >15 at initialisation of the simulation. The result is returned as a rounded integer.
    """
    params = self.parameters
    result = pd.Series(data=params['intercept_parity_lr2010'], index=df.index)

    result += (df.age_years * params['effect_age_parity_lr2010'])

    result[df.li_mar_stat == 2] += params['effect_mar_stat_2_parity_lr2010']
    result[df.li_mar_stat == 3] += params['effect_mar_stat_3_parity_lr2010']
    result[df.li_wealth == 1] += params['effect_wealth_lev_1_parity_lr2010']
    result[df.li_wealth == 2] += params['effect_wealth_lev_2_parity_lr2010']
    result[df.li_wealth == 3] += params['effect_wealth_lev_3_parity_lr2010']
    result[df.li_wealth == 4] += params['effect_wealth_lev_4_parity_lr2010']

    result[df.li_ed_lev == 2] += params['effect_edu_lev_2_parity_lr2010']
    result[df.li_ed_lev == 3] += params['effect_edu_lev_3_parity_lr2010']

    result[~df.li_urban] += params['effect_rural_parity_lr2010']

    # Return the result as a rounded integer (values are originally floats and can be negative)
    result = result.round()
    result.loc[result < 0] = 0
    result = result.astype(int)

    return result


def predict_obstruction_cpd_ip(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing obstructed labour
    secondary to cephalopelvic disproportion. Risk is increased in women who were extremely stunted as children and in
     women whose foetus is macrosomic (>4kg)
    """
    person = df.iloc[0]
    params = self.module.current_parameters
    result = params['prob_obstruction_cpd']

    # Effect of stunting only applied if the module is registered
    if 'Stunting' in self.module.sim.modules:
        if person['un_HAZ_category'] != 'HAZ>=-2':
            result *= params['rr_obstruction_cpd_stunted_mother']

    if externals['macrosomia']:
        result *= params['rr_obstruction_foetal_macrosomia']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_chorioamnionitis_ip(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing sepsis secondary
    to chorioamnionitis during labour. Risk is increased in women with premature rupture of membranes (PROM) but is
    decreased for women who have received antibiotics for PROM and in those who experienced clean delivery practices
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_sepsis_chorioamnionitis']

    if person['ps_premature_rupture_of_membranes']:
        result *= params['rr_sepsis_chorio_prom']
    if person['ac_received_abx_for_prom']:
        result *= params['treatment_effect_maternal_chorio_abx_prom']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_endometritis_pp(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing sepsis secondary
    to endometritis during the postnatal period. Risk is increased in women who delivered via caesarean section and
    decreased in those who experienced clean delivery practices
    """
    params = self.parameters
    result = params['prob_sepsis_endometritis']

    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['rr_sepsis_endometritis_post_cs']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_skin_soft_tissue_pp(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing sepsis secondary
    to skin/soft tissue infection during the postnatal period.Risk is increased in women who delivered via caesarean
    section and decreased in those who experienced clean delivery practices
    """
    params = self.parameters
    result = params['prob_sepsis_skin_soft_tissue']

    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['rr_sepsis_sst_post_cs']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_urinary_tract_pp(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing sepsis secondary
    to urinary tract infection during the postnatal period. Risk is decreased in those who experienced clean
    delivery practices
    """
    params = self.parameters
    result = params['prob_sepsis_urinary_tract']

    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to postpartum sepsis.
    Probability of death is reduced in the presence of treatment.
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_pp_sepsis']

    if person['la_sepsis_treatment']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'sepsis_treatment_effect_md', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_eclampsia_death(self, df, rng=None, **externals):
    """
    This is an individual level linear model which predicts an individuals probability of death due to eclampsia.
    Probability of death is reduced in the presence of treatment. Separate treatment effects are present for
    magnesium sulphate and IV antihypertensives
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_eclampsia']

    if person['la_eclampsia_treatment'] or person['ac_mag_sulph_treatment']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'eclampsia_treatment_effect_md', params)

        result *= treatment_effect

    if person['la_maternal_hypertension_treatment'] or person['ac_iv_anti_htn_treatment']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'anti_htns_treatment_effect_md', params)

        result *= treatment_effect

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_severe_pre_eclamp_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to severe
    pre-eclampsia. Probability of death is reduced in the presence of treatment.
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_severe_pre_eclamp']

    if person['la_maternal_hypertension_treatment'] or person['ac_iv_anti_htn_treatment']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'anti_htns_treatment_effect_md', params)

        result *= treatment_effect

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_placental_abruption_ip(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing placental
    abruption during labour. Risk is increased in women who have previously delivered
    via caesarean section and in women with gestational hypertension
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_placental_abruption_during_labour']

    if person['la_previous_cs_delivery']:
        result *= params['rr_placental_abruption_previous_cs']
    if person['ps_htn_disorders'] != 'none':
        result *= params['rr_placental_abruption_hypertension']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_ip(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing an antepartum
    haemorrhage during labour. We assume no risk of bleeding in the absence of
    predictors included in the model. We therefore use an additive approach to determine risk of bleeding in the
    presence of either placental abruption or placenta praevia
    """
    person = df.iloc[0]
    params = self.parameters
    result = 0.0

    if person['ps_placenta_praevia']:
        result += params['prob_aph_placenta_praevia_labour']
    if person['ps_placental_abruption'] or person['la_placental_abruption']:
        result += params['prob_aph_placental_abruption_labour']

    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to antepartum
    haemorrhage. Probability of death is reduced in the presence of treatment. Separate treatment effects are present
    for blood transfusion and caesarean delivery
    """
    params = self.parameters
    result = params['cfr_aph']

    if externals['received_blood_transfusion']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'aph_bt_treatment_effect_md', params)

        result *= treatment_effect

    if externals['mode_of_delivery'] == 'caesarean_section':
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'aph_cs_treatment_effect_md', params)

        result *= treatment_effect

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_pph_uterine_atony_pp(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing a postpartum haemorrhage due
    to an atonic uterus following birth. Risk is increased due to hypertension and decreased in the presence of
     prophylactic treatment
    """
    person = df.iloc[0]
    params = self.module.current_parameters
    result = params['prob_pph_uterine_atony']

    if externals['amtsl_given']:
        result *= params['treatment_effect_amtsl']

    if person['pn_htn_disorders'] != 'none':
        result *= params['rr_pph_ua_hypertension']

    if 'CardioMetabolicDisorders' in self.module.sim.modules:
        if (person['pn_htn_disorders'] == 'none') and (person['nc_hypertension']):
            result *= params['rr_pph_ua_hypertension']

    if person['ps_multiple_pregnancy']:
        result *= params['rr_pph_ua_multiple_pregnancy']
    if person['la_placental_abruption'] or person['ps_placental_abruption']:
        result *= params['rr_pph_ua_placental_abruption']
    if externals['macrosomia']:
        result *= params['rr_pph_ua_macrosomia']

    return pd.Series(data=[result], index=df.index)


def predict_pph_retained_placenta_pp(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing a postpartum
    haemorrhage due to an retained placenta following birth. Risk reduced by the presence of
    prophylactic treatment
    """
    params = self.parameters
    result = params['prob_pph_retained_placenta']

    if externals['amtsl_given']:
        result *= params['treatment_effect_amtsl']

    return pd.Series(data=[result], index=df.index)


def predict_postpartum_haem_pp_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death following a postpartum
    haemorrhage. Probability of death is reduced in the presence of treatment and increased in women with anaemia.
    Seperate treatment effects are present for surgical management or blood transfusion
    """
    person = df.iloc[0]
    params = self.module.current_parameters
    treatment = self.module.pph_treatment.to_strings(person.la_postpartum_haem_treatment)
    result = params['cfr_pp_pph']

    if ('surgery' in treatment) or ('hysterectomy' in treatment):  # todo: replace bitset property with bool?
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'pph_treatment_effect_surg_md', params)

        result *= treatment_effect

    if externals['received_blood_transfusion']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'pph_bt_treatment_effect_md', params)

        result *= treatment_effect

    if person['ps_anaemia_in_pregnancy'] or person['pn_anaemia_following_pregnancy']:
        result *= params['rr_pph_death_anaemia']

    return pd.Series(data=[result], index=df.index)


def predict_uterine_rupture_ip(self, df, rng=None, **externals):
    """
    Population level linear model to allow for the model to be scaled at initialisation of the simulation. The model
    returns a df containing the probability of developing a uterine rupture during labour/delivery. Risk is increased
    for women of greater parity, those who've delivered previously via caesarean section and women in obstructed labour
    """
    params = self.parameters
    result = pd.Series(data=params['prob_uterine_rupture'], index=df.index)

    result[df.la_parity == 2] *= params['rr_ur_parity_2']
    result[(df.la_parity > 2) & (df.la_parity < 5)] *= params['rr_ur_parity_3_or_4']
    result[df.la_parity >= 5] *= params['rr_ur_parity_5+']

    result[df.la_previous_cs_delivery] *= params['rr_ur_prev_cs']
    result[df.la_obstructed_labour] *= params['rr_ur_obstructed_labour']

    return result


def predict_uterine_rupture_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death following a uterine
    rupture. Probability of death is reduced in the presence of treatment. Separate treatment effects represent surgical
    interventions and blood transfusion
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_uterine_rupture']

    if person['la_uterine_rupture_treatment'] or person['la_has_had_hysterectomy']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'ur_repair_treatment_effect_md', params)

        result *= treatment_effect

    if externals['received_blood_transfusion']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'ur_treatment_effect_bt_md', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_intrapartum_still_birth(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of experiencing an intrapartum
    stillbirth during labour/delivery. Risk is increased due to maternal death, uterine rupture, obstructed labour,
    haemorrhage, hypertension, infection, twin pregnancy and is decreased in women delivering via caesarean section or
    assisted vaginal delivery
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_ip_still_birth']

    if not person['is_alive']:
        result *= params['rr_still_birth_maternal_death']
    if person['la_uterine_rupture']:
        result *= params['rr_still_birth_ur']
    if person['la_obstructed_labour']:
        result *= params['rr_still_birth_ol']
    if person['la_antepartum_haem'] != 'none':
        result *= params['rr_still_birth_aph']
    if person['ps_antepartum_haemorrhage'] != 'none':
        result *= params['rr_still_birth_aph']

    if person['ps_htn_disorders'] != 'none':
        result *= params['rr_still_birth_hypertension']

    if person['la_sepsis'] or person['ps_chorioamnionitis']:
        result *= params['rr_still_birth_sepsis']
    if person['ps_multiple_pregnancy']:
        result *= params['rr_still_birth_multiple_pregnancy']

    if externals['mode_of_delivery'] == 'caesarean_section':
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'treatment_effect_cs_still_birth', params)

        result *= treatment_effect

    if externals['mode_of_delivery'] == 'instrumental':
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            externals['delay_one_two'], externals['delay_three'], 'treatment_effect_avd_still_birth', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_probability_delivery_health_centre(self, df, rng=None, **externals):
    """
    Population level to allow for scaling at the initialisation of the simulation. This model predicts an
    individuals probability of seeking delivery care in a health centre instead of a hospital. This model is logistic
    as the parameters are odds and odds ratios and the result is returned as a probability. Odds of delivering in a
    health centre are decreased as women get older, decreased as women get richer, increased at higher parity,
    increased in married women and rural women
    """
    params = self.parameters
    result = pd.Series(data=params['odds_deliver_in_health_centre'], index=df.index)

    result[(df.age_years > 19) & (df.age_years < 25)] *= params['rrr_hc_delivery_age_20_24']
    result[(df.age_years > 24) & (df.age_years < 30)] *= params['rrr_hc_delivery_age_25_29']
    result[(df.age_years > 29) & (df.age_years < 35)] *= params['rrr_hc_delivery_age_30_34']
    result[(df.age_years > 34) & (df.age_years < 40)] *= params['rrr_hc_delivery_age_35_39']
    result[(df.age_years > 39) & (df.age_years < 45)] *= params['rrr_hc_delivery_age_40_44']
    result[(df.age_years > 44) & (df.age_years < 50)] *= params['rrr_hc_delivery_age_45_49']

    result[df.li_wealth == 1] *= params['rrr_hc_delivery_wealth_1']
    result[df.li_wealth == 2] *= params['rrr_hc_delivery_wealth_2']
    result[df.li_wealth == 3] *= params['rrr_hc_delivery_wealth_3']
    result[df.li_wealth == 4] *= params['rrr_hc_delivery_wealth_4']

    result[(df.la_parity > 2) & (df.la_parity < 5)] *= params['rrr_hc_delivery_parity_3_to_4']
    result[(df.la_parity > 4)] *= params['rrr_hc_delivery_parity_>4']

    result[~df.li_urban] *= params['rrr_hc_delivery_rural']

    result[df.li_mar_stat == 2] *= params['rrr_hc_delivery_married']

    result = result / (1 + result)
    return result


def predict_probability_delivery_at_home(self, df, rng=None, **externals):
    """
    Population level to allow for scaling at the initialisation of the simulation. This model predicts an
    individuals probability of undergoing a home birth  instead of a hospital. This model is logistic
    as the parameters are odds and odds ratios and the result is returned as a probability. Odds of homebirth are
     increased in younger women and decreased in older women, are higher in rural women, decreased in women with primary
     or secondary education, decreased in higher wealth and increased in higher parity
    """
    params = self.parameters
    result = pd.Series(data=params['odds_deliver_at_home'], index=df.index)

    result[(df.age_years > 19) & (df.age_years < 25)] *= params['rrr_hb_delivery_age_20_24']
    result[(df.age_years > 24) & (df.age_years < 30)] *= params['rrr_hb_delivery_age_25_29']
    result[(df.age_years > 29) & (df.age_years < 35)] *= params['rrr_hb_delivery_age_30_34']
    result[(df.age_years > 34) & (df.age_years < 40)] *= params['rrr_hb_delivery_age_35_39']
    result[(df.age_years > 39) & (df.age_years < 45)] *= params['rrr_hb_delivery_age_40_44']
    result[(df.age_years > 44) & (df.age_years < 50)] *= params['rrr_hb_delivery_age_45_49']

    result[~df.li_urban] *= params['rrr_hb_delivery_rural']

    result[(df.la_parity > 2) & (df.la_parity < 5)] *= params['rrr_hb_delivery_parity_3_to_4']
    result[(df.la_parity > 4)] *= params['rrr_hb_delivery_parity_>4']

    result[df.li_ed_lev == 2] *= params['rrr_hb_delivery_primary_education']
    result[df.li_ed_lev == 3] *= params['rrr_hb_delivery_secondary_education']

    result[df.li_wealth == 1] *= params['rrr_hb_delivery_wealth_1']
    result[df.li_wealth == 2] *= params['rrr_hb_delivery_wealth_2']
    result[df.li_wealth == 3] *= params['rrr_hb_delivery_wealth_3']
    result[df.li_wealth == 4] *= params['rrr_hb_delivery_wealth_4']

    result[df.li_mar_stat == 3] *= params['rrr_hb_delivery_married']

    result = result / (1 + result)
    return result


def predict_postnatal_check(self, df, rng=None, **externals):
    """
    Population level to allow for scaling at the initialisation of the simulation. This model predicts an
    individuals probability of receiving postnatal care following their delivery. This model is logistic
    as the parameters are odds and odds ratios and the result is returned as a probability. Odds of PNC are higher in
    older women, lower in rural women, lower in the poorest women, higher in women with ANC4+, higher in those
    delivering in a facility and via caeasrean section and lower in women with higher parity
    """
    params = self.parameters
    result = pd.Series(data=params['odds_will_attend_pnc'], index=df.index)

    result[(df.age_years > 29) & (df.age_years < 36)] *= params['or_pnc_age_30_35']
    result[(df.age_years >= 36)] *= params['or_pnc_age_>35']

    result[~df.li_urban] *= params['or_pnc_rural']

    result[df.li_wealth == 1] *= params['or_pnc_wealth_level_1']

    result[(df.la_parity > 4)] *= params['or_pnc_parity_>4']

    result[(df.ac_total_anc_visits_current_pregnancy > 3)] *= params['or_pnc_anc4+']

    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['or_pnc_caesarean_delivery']
    result[(externals['delivery_setting'] == 'health_centre') | (externals['delivery_setting'] == 'hospital')] \
        *= params['or_pnc_facility_delivery']

    result = result / (1 + result)
    return result
