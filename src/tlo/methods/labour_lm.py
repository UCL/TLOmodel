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


def predict_parity(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['intercept_parity_lr2010'], index=df.index)
    result += df.age_years * 0.22
    result[df.li_mar_stat == 2] += params['effect_mar_stat_2_parity_lr2010']
    result[df.li_mar_stat == 3] += params['effect_mar_stat_3_parity_lr2010']
    result += df.li_wealth.map(
        {
            1: params[f'effect_wealth_lev_1_parity_lr2010'],
            2: params[f'effect_wealth_lev_2_parity_lr2010'],
            3: params[f'effect_wealth_lev_3_parity_lr2010'],
            4: params[f'effect_wealth_lev_4_parity_lr2010'],
            5: params[f'effect_wealth_lev_5_parity_lr2010'],
        }
    )
    return result


def predict_obstructed_labour_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.cause_of_obstructed_labour.to_strings(person.la_obstructed_labour_causes)
    result = 0.0
    if 'cephalopelvic_dis' in causes:
        result += params['prob_obstruction_cpd']
    if 'malposition' in causes:
        result += params['prob_obstruction_malpos']
    if 'malpresentation' in causes:
        result += params['prob_obstruction_malpres']
    return pd.Series(data=[result], index=df.index)


def predict_chorioamnionitis_ip(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_chorioamnionitis_ip']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_other_maternal_infections_ip(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_other_maternal_infection_ip']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_endometritis_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_endometritis_pp']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_skin_soft_tissue_inf_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_skin_soft_tissue_inf_pp']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_urinary_tract_inf_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_urinary_tract_inf_pp']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_other_maternal_infection_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_other_maternal_infection_pp']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.intrapartum_infections.to_strings(person.la_maternal_ip_infection)
    result = 0.0

    if 'chorioamnionitis' in causes:
        result += params['prob_sepsis_chorioamnionitis']
    if 'other_maternal_infection' in causes:
        result += params['prob_sepsis_other_maternal_infection_ip']
    if person['ps_chorioamnionitis'] == 'histological':
        result += params['prob_sepsis_chorioamnionitis']
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_sepsis']

    if person['la_sepsis_treatment']:
        result *= params['sepsis_treatment_effect_md']
    if person['ac_received_abx_for_chorioamnionitis']:
        result *= 0.5  # todo: save as param
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_pp(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.postpartum_infections.to_strings(person.la_maternal_pp_infection)
    result = 0.0

    if 'endometritis' in causes:
        result += params['prob_sepsis_endometritis']
    if 'urinary_tract_inf' in causes:
        result += params['prob_sepsis_urinary_tract_inf']
    if 'skin_soft_tissue_inf' in causes:
        result += params['prob_sepsis_skin_soft_tissue_inf']
    if 'other_maternal_infection' in causes:
        result += params['prob_sepsis_other_maternal_infection_pp']

    return pd.Series(data=[result], index=df.index)


def predict_sepsis_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_sepsis']

    if person['la_sepsis_treatment']:
        result *= params['sepsis_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_eclampsia_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_eclampsia']

    if person['la_eclampsia_treatment']:
        result *= params['eclampsia_treatment_effect_md']
    # Both these predictors represent intravenous antihypertensives- both will not be true for the same
    # woman
    if person['la_maternal_hypertension_treatment']:
        result *= params['anti_htns_treatment_effect_md']
    if person['ac_iv_anti_htn_treatment']:
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_eclampsia_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_pp_eclampsia'] # todo: collapse as one eclampsia death equation

    if person['la_eclampsia_treatment']:
        result *= params['eclampsia_treatment_effect_md']
    # Both these predictors represent intravenous antihypertensives- both will not be true for the same
    # woman
    if person['la_maternal_hypertension_treatment']:
        result *= params['anti_htns_treatment_effect_md']
    if person['ac_iv_anti_htn_treatment']:
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_severe_pre_eclamp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_severe_pre_eclamp']

    if person['la_maternal_hypertension_treatment']:
        result *= params['anti_htns_treatment_effect_md']
    if person['ac_iv_anti_htn_treatment']:
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_placental_abruption_ip(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_placental_abruption_during_labour']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = 0.0

    if person['ps_placenta_praevia']:
        result += params['prob_aph_placenta_praevia_labour']
    if person['ps_placental_abruption']:
        result += params['prob_aph_placental_abruption_labour']
    if person['la_placental_abruption']:
        result += params['prob_aph_placental_abruption_labour']

    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_death(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['cfr_aph']

    if externals['received_blood_transfusion']:
        result *= params['aph_bt_treatment_effect_md']
    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['aph_cs_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_postpartum_haem_pp(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.cause_of_primary_pph.to_strings(person.la_postpartum_haem_cause)
    result = 0.0

    if 'uterine_atony' in causes:
        result += params['prob_pph_uterine_atony']
    if 'lacerations' in causes:
        result += params['prob_pph_lacerations']
    if 'retained_placenta' in causes:
        result += params['prob_pph_retained_placenta']
    if 'other_pph_cause' in causes:
        result += params['prob_pph_other_causes']

    return pd.Series(data=[result], index=df.index)


def predict_postpartum_haem_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.pph_treatment.to_strings(person.la_postpartum_haem_treatment)
    result = params['cfr_pp_pph']

    if 'uterotonics' in causes:
        result *= params['pph_treatment_effect_uterotonics_md']
    if 'manual_removal_placenta' in causes:
        result *= params['pph_treatment_effect_mrp_md']
    if 'surgery' in causes:
        result *= params['pph_treatment_effect_surg_md']
    if 'hysterectomy' in causes:
        result *= params['pph_treatment_effect_hyst_md']
    if externals['received_blood_transfusion']:
        result *= params['pph_bt_treatment_effect_md']
    if person['ps_anaemia_in_pregnancy']:
        result *= params['rr_pph_death_anaemia']

    return pd.Series(data=[result], index=df.index)


def predict_uterine_rupture_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['odds_uterine_rupture']

    if person['la_parity'] > 4:
        result *= params['or_ur_grand_multip']
    if person['la_previous_cs_delivery']:
        result *= params['or_ur_prev_cs']
    if person['la_obstructed_labour']:
        result *= params['or_ur_ref_ol']

    # convert back to probability
    result = result / (1 + result)

    return pd.Series(data=[result], index=df.index)


def predict_uterine_rupture_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_uterine_rupture']

    if person['la_uterine_rupture_treatment'] > 4:
        result *= params['ur_repair_treatment_effect_md']
    if person['la_has_had_hysterectomy']:
        result *= params['or_ur_prev_cs']
    if externals['received_blood_transfusion']:
        result *= params['ur_treatment_effect_bt_md']

    return pd.Series(data=[result], index=df.index)


def predict_intrapartum_still_birth(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_ip_still_birth_unk_cause']

    if person['la_maternal_death_in_labour']:
        result *= params['rr_still_birth_maternal_death']
    if person['la_antepartum_haem'] != 'none':
        result *= params['rr_still_birth_aph']
    # todo: risk should modify with severity- placeholder

    if person['ps_antepartum_haemorrhage'] != 'none':
        result *= params['rr_still_birth_aph']
    if person['la_obstructed_labour']:
        result *= params['rr_still_birth_ol']
    if person['la_uterine_rupture']:
        result *= params['rr_still_birth_ur']
    if person['la_sepsis']:
        result *= params['rr_still_birth_sepsis']
    # todo: chorio

    if person['ps_htn_disorders'] == 'severe_pre_eclamp':
        result *= params['rr_still_birth_spe']
    if person['ps_htn_disorders'] == 'eclampsia':
        result *= params['rr_still_birth_ec']
    if externals['mode_of_delivery'] == 'instrumental':
        result *= params['treatment_effect_avd_still_birth']
    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['treatment_effect_cs_still_birth']

    return pd.Series(data=[result], index=df.index)


def predict_probability_delivery_health_centre(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['odds_deliver_in_health_centre']

    # todo: should this be logistic?
    if 24 < person['age_years'] < 30:
        result *= params['rrr_hc_delivery_age_25_29']
    if 29 < person['age_years'] < 35:
        result *= params['rrr_hc_delivery_age_30_34']
    if 34 < person['age_years'] < 40:
        result *= params['rrr_hc_delivery_age_35_39']
    if 39 < person['age_years'] < 45:
        result *= params['rrr_hc_delivery_age_40_44']
    if 44 < person['age_years'] < 50:
        result *= params['rrr_hc_delivery_age_45_49']

    if ~person['li_urban']:
        result *= params['rrr_hc_delivery_rural']
    if 2 < person['la_parity'] < 5:
        result *= params['rrr_hc_delivery_parity_3_to_4']
    if person['la_parity'] > 4:
        result *= params['rrr_hc_delivery_parity_>4']
    if person['li_mar_stat'] == 2:
        result *= params['rrr_hc_delivery_married']

    return pd.Series(data=[result], index=df.index)


def predict_probability_delivery_at_home(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['odds_deliver_at_home']

    # todo: should this be logistic?
    if 34 < person['age_years'] < 40:
        result *= params['rrr_hb_delivery_age_35_39']
    if 39 < person['age_years'] < 45:
        result *= params['rrr_hb_delivery_age_40_44']
    if 44 < person['age_years'] < 50:
        result *= params['rrr_hb_delivery_age_45_49']

    if 2 < person['la_parity'] < 5:
        result *= params['rrr_hb_delivery_parity_3_to_4']
    if person['la_parity'] > 4:
        result *= params['rrr_hb_delivery_parity_>4']

    return pd.Series(data=[result], index=df.index)


def predict_care_seeking_for_complication(self, df, rng=None, **externals):
    """individual level"""
    #  person = df.iloc[0]
    params = self.parameters
    result = params['prob_careseeking_for_complication']

    return pd.Series(data=[result], index=df.index)
