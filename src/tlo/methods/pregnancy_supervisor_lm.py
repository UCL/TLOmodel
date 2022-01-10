import pandas as pd


def preterm_labour(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=1, index=df.index)

    result[df.ps_gestational_age_in_weeks == 22] *= params['baseline_prob_early_labour_onset'][0]
    result[df.ps_gestational_age_in_weeks == 27] *= params['baseline_prob_early_labour_onset'][1]
    result[df.ps_gestational_age_in_weeks == 31] *= params['baseline_prob_early_labour_onset'][2]
    result[df.ps_gestational_age_in_weeks == 35] *= params['baseline_prob_early_labour_onset'][3]

    result[df.ps_premature_rupture_of_membranes] *= params['rr_preterm_labour_post_prom']
    result[df.ps_anaemia_in_pregnancy != 'none'] *= params['rr_preterm_labour_anaemia']
    result[df.ma_is_infected] *= params['rr_preterm_labour_malaria']
    result[df.ps_multiple_pregnancy] *= params['rr_preterm_labour_multiple_pregnancy']

    return result


def placenta_praevia(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_placenta_praevia'], index=df.index)

    result[df.la_previous_cs_delivery] *= params['rr_placenta_praevia_previous_cs']

    return result


def antepartum_haem(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=0, index=df.index)

    result[df.ps_placenta_praevia] += params['prob_aph_placenta_praevia']
    result[df.ps_placental_abruption] += params['prob_aph_placental_abruption']

    return result


def ectopic_pregnancy_death(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_ectopic_pregnancy_death'], index=df.index)

    result[df.ac_ectopic_pregnancy_treated] *= params['treatment_effect_ectopic_pregnancy_treatment']

    return result


def induced_abortion_death(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_induced_abortion_death'], index=df.index)

    result[df.ac_post_abortion_care_interventions > 0] *= params['treatment_effect_post_abortion_care']

    return result


def spontaneous_abortion(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=1, index=df.index)

    result[df.ps_gestational_age_in_weeks == 4] *= params['prob_spontaneous_abortion_per_month'][0]
    result[df.ps_gestational_age_in_weeks == 8] *= params['prob_spontaneous_abortion_per_month'][1]
    result[df.ps_gestational_age_in_weeks == 13] *= params['prob_spontaneous_abortion_per_month'][2]
    result[df.ps_gestational_age_in_weeks == 17] *= params['prob_spontaneous_abortion_per_month'][3]
    result[df.ps_gestational_age_in_weeks == 22] *= params['prob_spontaneous_abortion_per_month'][4]

    result[df.ps_prev_spont_abortion] *= params['rr_spont_abortion_prev_sa']

    result[(df.age_years > 29) & (df.age_years < 35)] *= params['rr_spont_abortion_age_31_34']
    result[df.age_years > 34] *= params['rr_spont_abortion_age_35']

    return result


def spontaneous_abortion_death(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_spontaneous_abortion_death'], index=df.index)

    result[df.ac_post_abortion_care_interventions > 0] *= params['treatment_effect_post_abortion_care']

    return result


def maternal_anaemia(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['baseline_prob_anaemia_per_month'], index=df.index)

    result[df.ma_is_infected] *= params['rr_anaemia_maternal_malaria']
    result[df.hv_inf & (df.hv_art != 'not')] *= params['rr_anaemia_maternal_malaria']
    result[df.ac_receiving_iron_folic_acid] *= params['treatment_effect_iron_folic_acid_anaemia']

    return result


def gest_diab(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_gest_diab_per_month'], index=df.index)

    result[df.li_bmi > 3] *= params['rr_gest_diab_obesity']

    return result


def gest_htn(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_gest_htn_per_month'], index=df.index)

    result[df.li_bmi > 3] *= params['rr_gest_htn_obesity']
    result[df.ac_receiving_calcium_supplements] *= params['treatment_effect_gest_htn_calcium']

    return result


def pre_eclampsia(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_pre_eclampsia_per_month'], index=df.index)

    result[df.li_bmi > 3] *= params['rr_pre_eclampsia_obesity']
    result[df.ps_multiple_pregnancy] *= params['rr_pre_eclampsia_multiple_pregnancy']
    result[df.nc_hypertension] *= params['rr_pre_eclampsia_chronic_htn']
    result[df.nc_diabetes] *= params['rr_pre_eclampsia_diabetes_mellitus']
    result[df.ac_receiving_calcium_supplements] *= params['treatment_effect_calcium_pre_eclamp']

    return result


def placental_abruption(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_placental_abruption_per_month'], index=df.index)

    result[df.la_previous_cs_delivery] *= params['rr_placental_abruption_previous_cs']
    result[df.ps_htn_disorders != 'none'] *= params['rr_placental_abruption_hypertension']

    return result


def antenatal_stillbirth(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_still_birth_per_month'], index=df.index)

    result[df.ps_gestational_age_in_weeks == 41] *= params['rr_still_birth_ga_41']
    result[df.ps_gestational_age_in_weeks == 42] *= params['rr_still_birth_ga_42']
    result[df.ps_gestational_age_in_weeks > 42] *= params['rr_still_birth_ga_>42']

    result[df.ps_htn_disorders == 'mild_pre_eclamp'] *= params['rr_still_birth_pre_eclampsia']
    result[df.ps_htn_disorders == 'severe_pre_eclamp'] *= params['rr_still_birth_pre_eclampsia']
    result[df.ps_htn_disorders == 'gest_htn'] *= params['rr_still_birth_gest_htn']
    result[df.ps_htn_disorders == 'severe_gest_htn'] *= params['rr_still_birth_gest_htn']

    result[df.ps_antepartum_haemorrhage != 'none'] *= params['rr_still_birth_aph']
    result[df.ps_chorioamnionitis] *= params['rr_still_birth_chorio']
    result[df.nc_hypertension] *= params['rr_still_birth_chronic_htn']
    result[(df.ps_gest_diab == 'controlled') & (df.ac_gest_diab_on_treatment != 'none')] *=\
        (params['rr_still_birth_gest_diab'] * params['treatment_effect_gdm_case_management'])

    result[df.ma_is_infected] *= params['rr_still_birth_maternal_malaria']
    result[df.nc_diabetes] *= params['rr_still_birth_diab_mellitus']
    result[df.ps_syphilis] *= params['rr_still_birth_maternal_syphilis']

    return result


def early_initiation_anc4(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['odds_early_init_anc4'], index=df.index)

    result[(df.age_years > 19) & (df.age_years < 25)] *= params['aor_early_anc4_20_24']
    result[(df.age_years > 24) & (df.age_years < 30)] *= params['aor_early_anc4_25_29']
    result[(df.age_years > 29) & (df.age_years < 35)] *= params['aor_early_anc4_30_34']
    result[(df.age_years > 34) & (df.age_years < 40)] *= params['aor_early_anc4_35_39']
    result[(df.age_years > 39) & (df.age_years < 45)] *= params['aor_early_anc4_40_44']
    result[(df.age_years > 44) & (df.age_years < 50)] *= params['aor_early_anc4_45_49']

    if externals['year'] < 2015:
        result *= params['aor_early_anc4_2010']
    else:
        result *= params['aor_early_anc4_2015']

    result[(df.la_parity > 1) & (df.la_parity < 4)] *= params['aor_early_anc4_parity_2_3']
    result[(df.la_parity > 3) & (df.la_parity < 6)] *= params['aor_early_anc4_parity_4_5']
    result[df.la_parity > 5] *= params['aor_early_anc4_parity_6+']

    result[df.li_ed_lev == 2] *= params['aor_early_anc4_primary_edu']
    result[df.li_ed_lev == 3] *= params['aor_early_anc4_secondary_edu']

    result[df.li_wealth == 1] *= params['aor_early_anc4_richest_wealth']
    result[df.li_wealth == 2] *= params['aor_early_anc4_richer_wealth']
    result[df.li_wealth == 3] *= params['aor_early_anc4_middle_wealth']

    result[df.li_mar_stat == 2] *= params['aor_early_anc4_married']
    result[df.li_mar_stat == 3] *= params['aor_early_anc4_previously_married']

    result = result / (1 + result)
    return result
