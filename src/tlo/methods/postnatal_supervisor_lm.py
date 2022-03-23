import pandas as pd


def predict_obstetric_fistula(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing an obstetric fistula
    following delivery. Risk increased for women who experience obstructed labour
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_obstetric_fistula']

    if person['la_obstructed_labour']:
        result *= params['rr_obstetric_fistula_obstructed_labour']

    return pd.Series(data=[result], index=df.index)


def predict_secondary_postpartum_haem(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing an secondary postpartum
    haemorrhage following delivery. Risk increased for women who experience endometritis
    """
    params = self.parameters
    result = pd.Series(data=params['prob_secondary_pph'], index=df.index)

    result[externals['endometritis']] *= params['rr_secondary_pph_endometritis']

    return result


def predict_sepsis_endometritis_late_postpartum(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing sepsis secondary to
    endometritis. Risk increased for women who deliver via caesarean
    """
    params = self.parameters
    result = pd.Series(data=params['prob_late_sepsis_endometritis'], index=df.index)

    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['rr_sepsis_endometritis_post_cs']

    return result


def predict_sepsis_sst_late_postpartum(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing sepsis secondary to
    skin/soft tissue infection. Risk increased for women who deliver via caesarean
    """
    params = self.parameters
    result = pd.Series(data=params['prob_late_sepsis_skin_soft_tissue'], index=df.index)

    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['rr_sepsis_sst_post_cs']

    return result


def predict_gest_htn_pn(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing postpartum gestational
    hypertension. Risk increased for women who are obese
    """
    params = self.parameters
    result = pd.Series(data=params['weekly_prob_gest_htn_pn'], index=df.index)

    result[df.li_bmi > 3] *= params['rr_gest_htn_obesity']

    return result


def predict_pre_eclampsia_pn(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing postpartum pre-eclampsia.
    Risk increased for women who are obese, experience non-gestational hypertension and diabetes mellitus
    """
    params = self.module.current_parameters
    result = pd.Series(data=params['weekly_prob_pre_eclampsia_pn'], index=df.index)

    result[df.li_bmi > 3] *= params['rr_gest_htn_obesity']

    if 'CardioMetabolicDisorders' in self.module.sim.modules:
        result[df.nc_hypertension] *= params['rr_pre_eclampsia_chronic_htn']
        result[df.nc_diabetes] *= params['rr_pre_eclampsia_diabetes_mellitus']

    return result


def predict_anaemia_after_pregnancy(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing postpartum anaemia.
    Risk increased for women who have malaria or untreated HIV and is decreased if receiving daily iron
    """
    m = self.module
    p = m.current_parameters

    result = pd.Series(data=p['baseline_prob_anaemia_per_week'], index=df.index)
    result[df.la_iron_folic_acid_postnatal] *= p['treatment_effect_iron_folic_acid_anaemia']

    if 'Hiv' in self.module.sim.modules:
        result[df.hv_inf & (df.hv_art != "not")] *= p['rr_anaemia_hiv_no_art']

    if 'Malaria' in self.module.sim.modules:
        result[df.ma_is_infected] *= p['rr_anaemia_maternal_malaria']

    return result


def predict_early_onset_neonatal_sepsis_week_1(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing early onset neonatal sepsis.
    Risk increased for newborns of mothers with chorioamnionitis, PROM or preterm neonates. Risk is decreased following
    antibiotic therapy for PROM, clean birth practices, cord care and early initiation of breastfeeding
    """
    params = self.parameters
    result = pd.Series(data=params['prob_early_onset_neonatal_sepsis_week_1'], index=df.index)

    result[externals['maternal_chorioamnionitis']] *= params['rr_eons_maternal_chorio']
    result[externals['maternal_prom']] *= params['rr_eons_maternal_prom']
    result[df.nb_early_preterm] *= params['rr_eons_preterm_neonate']
    result[df.nb_late_preterm] *= params['rr_eons_preterm_neonate']

    result[externals['received_abx_for_prom']] *= params['treatment_effect_abx_prom']
    result[df.nb_clean_birth] *= params['treatment_effect_clean_birth']
    result[df.nb_received_cord_care] *= params['treatment_effect_cord_care']
    result[df.nb_early_init_breastfeeding] *= params['treatment_effect_early_init_bf']

    return result


def predict_late_onset_neonatal_sepsis(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of developing late onset neonatal sepsis.
    Risk is decreased following early initiation of breastfeeding
    """
    params = self.parameters
    result = pd.Series(data=params['prob_late_onset_neonatal_sepsis'], index=df.index)

    result[df.nb_early_init_breastfeeding] *= params['treatment_effect_early_init_bf']

    return result


def predict_care_seeking_for_fistula_repair(self, df, rng=None, **externals):
    """
    Population level linear model which predicts an individuals probability of seeking care for treatment of an
    obstetric fistula. Risk is decreased in young age and lower education
    """
    params = self.parameters
    result = pd.Series(data=params['odds_care_seeking_fistula_repair'], index=df.index)

    result[df.age_years.between(14, 20)] *= params['aor_cs_fistula_age_15_19']
    result[df.li_ed_lev == 1] *= params['aor_cs_fistula_age_lowest_education']

    result = result / (1 + result)
    return result
