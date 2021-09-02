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


def predict_early_onset_neonatal_sepsis(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters

    result = params['prob_early_onset_neonatal_sepsis_day_0']

    if externals['maternal_chorioamnionitis']:
        result *= params['rr_eons_maternal_chorio']
    if person['ps_premature_rupture_of_membranes']:
        result *= params['rr_eons_maternal_prom']
    if person['nb_early_preterm'] or person['nb_late_preterm']:
        result *= params['rr_eons_preterm_neonate']

    if person['nb_clean_birth']:
        result *= params['treatment_effect_clean_birth']
    #if person['nb_received_cord_care']:   # TODO: currently this only effects later risk of sepsis
    #    result *= params['treatment_effect_cord_care']
    if person['nb_early_init_breastfeeding']:
        result *= params['treatment_effect_early_init_bf']
    if externals['received_abx_for_prom']:
        result *= params['treatment_effect_abx_prom']

    return pd.Series(data=[result], index=df.index)


def predict_encephalopathy(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    mother_id = person['mother_id']
    params = self.parameters
    main_df = self.module.sim.population.props

    result = params['prob_encephalopathy']

    if person['nb_early_onset_neonatal_sepsis']:
        result *= params['rr_enceph_neonatal_sepsis']
    if main_df.at[mother_id, 'la_obstructed_labour']:
        result *= params['rr_enceph_obstructed_labour']
    if main_df.at[mother_id, 'la_uterine_rupture'] or (main_df.at[mother_id, 'la_antepartum_haem'] != 'none') or\
        (main_df.at[mother_id, 'ps_antepartum_haemorrhage'] != 'none'):
        result *= params['rr_enceph_acute_hypoxic_event']

    return pd.Series(data=[result], index=df.index)


def predict_rds_preterm(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    mother_id = person['mother_id']
    params = self.parameters
    main_df = self.module.sim.population.props

    result = params['prob_respiratory_distress_preterm']

    if main_df.at[mother_id, 'nc_diabetes']:
        result *= params['rr_rds_maternal_diabetes_mellitus']
    if main_df.at[mother_id, 'ps_gest_diab'] != 'none':
        result *= params['rr_rds_maternal_gestational_diab']
    if externals['received_corticosteroids']:
        result *= params['treatment_effect_steroid_preterm']

    return pd.Series(data=[result], index=df.index)


def predict_not_breathing_at_birth(self, df, rng=None, **externals):
    """individual level"""
    # person = df.iloc[0]
    params = self.parameters
    result = params['prob_failure_to_transition']

    return pd.Series(data=[result], index=df.index)


def predict_retinopathy(self, df, rng=None, **externals):
    """individual level"""
    # person = df.iloc[0]
    params = self.parameters
    result = params['prob_retinopathy_preterm']

    return pd.Series(data=[result], index=df.index)


def predict_care_seeking_for_complication(self, df, rng=None, **externals):
    """individual level"""
    # person = df.iloc[0]
    params = self.parameters
    result = params['prob_care_seeking_for_complication']

    return pd.Series(data=[result], index=df.index)


def predict_preterm_birth_other_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_preterm_birth']

    if person['nb_early_preterm']:
        result *= params['rr_preterm_death_early_preterm']
    if person['nb_kangaroo_mother_care']:  # TODO: and low birth weight?
        result *= params['treatment_effect_kmc']

    return pd.Series(data=[result], index=df.index)


def predict_not_breathing_at_birth_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_failed_to_transition']

    if person['nb_received_neonatal_resus']:
        result *= params['treatment_effect_resuscitation']

    return pd.Series(data=[result], index=df.index)


def predict_enceph_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_enceph']

    if person['nb_encephalopathy'] == 'severe':
        result *= params['cfr_multiplier_severe_enceph']
    if person['nb_received_neonatal_resus']:
        result *= params['treatment_effect_resuscitation']

    return pd.Series(data=[result], index=df.index)


def predict_neonatal_sepsis_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_neonatal_sepsis']

    if person['nb_inj_abx_neonatal_sepsis']:
        result *= params['treatment_effect_inj_abx_sep']
    if person['nb_supp_care_neonatal_sepsis']:
        result *= params['treatment_effect_supp_care_sep']

    return pd.Series(data=[result], index=df.index)


def predict_respiratory_distress_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_respiratory_distress_syndrome']

    if person['nb_received_neonatal_resus']:
        result *= params['treatment_effect_resuscitation_preterm']

    return pd.Series(data=[result], index=df.index)
