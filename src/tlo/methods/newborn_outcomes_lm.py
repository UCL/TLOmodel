import pandas as pd
from tlo.methods import pregnancy_helper_functions

def predict_early_onset_neonatal_sepsis(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing early onset neonatal sepsis.
    Risk increased by the presence of maternal chorioamnionitis, PROM, or prematurity. Risk decreased by clean birth
    practices, early initiation of breastfeeding and antibiotic therapy for PROM
    """
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
    if person['nb_early_init_breastfeeding']:
        result *= params['treatment_effect_early_init_bf']
    if externals['received_abx_for_prom']:
        result *= params['treatment_effect_abx_prom']

    return pd.Series(data=[result], index=df.index)


def predict_encephalopathy(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing neonatal encephalopathy
    Risk increased by the presence of neonatal sepsis, maternal obstructed labour, maternal uterine rupture, or
    maternal antepartum haemorrhage
    """
    person = df.iloc[0]
    mother_id = person['mother_id']
    params = self.parameters
    main_df = self.module.sim.population.props

    result = params['prob_encephalopathy']

    if person['nb_early_onset_neonatal_sepsis']:
        result *= params['rr_enceph_neonatal_sepsis']
    if main_df.at[mother_id, 'la_obstructed_labour']:
        result *= params['rr_enceph_obstructed_labour']
    if main_df.at[mother_id, 'la_uterine_rupture'] or\
        (main_df.at[mother_id, 'la_antepartum_haem'] != 'none') or\
       (main_df.at[mother_id, 'ps_antepartum_haemorrhage'] != 'none'):
        result *= params['rr_enceph_acute_hypoxic_event']

    return pd.Series(data=[result], index=df.index)


def predict_rds_preterm(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of developing preterm respiratory distress
    syndrome.  Risk increased by the presence maternal gestational diabetes and diabetes mellitus. Risk decreased by
    steroid treatment
    """
    person = df.iloc[0]
    mother_id = person['mother_id']
    params = self.parameters
    main_df = self.module.sim.population.props

    result = params['prob_respiratory_distress_preterm']

    if main_df.at[mother_id, 'ps_gest_diab'] != 'none':
        result *= params['rr_rds_maternal_gestational_diab']
    if externals['received_corticosteroids']:
        result *= params['treatment_effect_steroid_preterm']

    if 'CardioMetabolicDisorders' in self.module.sim.modules:
        if main_df.at[mother_id, 'nc_diabetes']:
            result *= params['rr_rds_maternal_diabetes_mellitus']

    return pd.Series(data=[result], index=df.index)


def predict_preterm_birth_other_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to prematurity (other causes)
    syndrome.  Risk increased by in early preterm neonates and decreased following kangaroo mother care
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_preterm_birth']

    if person['nb_early_preterm']:
        result *= params['rr_preterm_death_early_preterm']
    if person['nb_kangaroo_mother_care']:
        result *= params['treatment_effect_kmc']

    return pd.Series(data=[result], index=df.index)


def predict_not_breathing_at_birth_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to not breathing at birth.
    Risk decreased by neonatal resuscitation
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_failed_to_transition']

    if person['nb_received_neonatal_resus']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            False, externals['delay'], 'treatment_effect_resuscitation', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_enceph_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to neonatal encephalopathy
    Risk decreased by neonatal resuscitation and increased in severe cases.
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_enceph']

    if person['nb_encephalopathy'] == 'severe_enceph':
        result *= params['cfr_multiplier_severe_enceph']

    if person['nb_received_neonatal_resus']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            False, externals['delay'], 'treatment_effect_resuscitation', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_neonatal_sepsis_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to neonatal sepsis.
    Risk decreased by treatment (either just antibiotics or full supportive care)
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_early_onset_sepsis']

    if person['nb_inj_abx_neonatal_sepsis']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            False, externals['delay'], 'treatment_effect_inj_abx_sep', params)

        result *= treatment_effect

    if person['nb_supp_care_neonatal_sepsis']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            False, externals['delay'], 'treatment_effect_supp_care_sep', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)


def predict_respiratory_distress_death(self, df, rng=None, **externals):
    """
    Individual level linear model which predicts an individuals probability of death due to preterm respiratory
    distress. Risk decreased by neonatal resuscitation
    """
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_respiratory_distress_syndrome']

    if person['nb_received_neonatal_resus']:
        treatment_effect = pregnancy_helper_functions.get_treatment_effect(
            False, externals['delay'], 'treatment_effect_resuscitation_preterm', params)

        result *= treatment_effect

    return pd.Series(data=[result], index=df.index)
