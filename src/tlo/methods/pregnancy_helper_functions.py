"""
This file stores any functions that are called from multiple different modules in the Maternal and Perinatal Health
module suite
"""

import pandas as pd
import numpy as np


def scale_linear_model_at_initialisation(self, model, parameter_key):
    """
    This function scales the intercept value of linear models according to the distribution of predictor values
    within the data frame. The parameter value (intercept of the model) is then updated accordingly
    :param model: model object to be scaled
    :param parameter_key: key (str) relating to the parameter which holds the target rate for the model
    """

    df = self.sim.population.props
    params = self.current_parameters

    # Select women and create dummy variable for externals called during model run
    women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mode_of_delivery = pd.Series(False, index=women.index)
    delivery_setting = pd.Series('none', index=women.index)

    # Create a function that runs a linear model with an intercept of 1 and generates a scaled intercept
    def return_scaled_intercept(target, logistic_model):
        mean = model.predict(women,
                             year=self.sim.date.year,
                             mode_of_delivery=mode_of_delivery,
                             delivery_setting=delivery_setting).mean()
        if logistic_model:
            mean = mean / (1.0 - mean)

        scaled_intercept = 1.0 * (target / mean) if (target != 0 and mean != 0 and not np.isnan(mean)) else 1.0
        return scaled_intercept

    # The target value is stored within the parameter
    target = params[parameter_key]

    if (parameter_key != 'prob_spontaneous_abortion_per_month') and (parameter_key != 'baseline_prob_early_'
                                                                                      'labour_onset'):

        # Override the intercept parameter with a value of one
        params[parameter_key] = 1

        # Function varies depending on the input/output of the model (i.e. if logistic)
        if 'odds' in parameter_key:
            params[parameter_key] = return_scaled_intercept(target, logistic_model=True)
        else:
            params[parameter_key] = return_scaled_intercept(target, logistic_model=False)

    else:
        # These models use predictor values dependent on gestational age to replace the intercept value
        # (see pregnancy_supervisor_lm.py) so we set those predictors to be one
        if parameter_key == 'prob_spontaneous_abortion_per_month':
            params[parameter_key] = [1, 1, 1, 1, 1]
        elif parameter_key == 'baseline_prob_early_labour_onset':
            params[parameter_key] = [1, 1, 1, 1]

        # And scale the values accordingly
        scaled_intercepts = list()
        for item in target:
            intercept = return_scaled_intercept(item, logistic_model=False)
            scaled_intercepts.append(intercept)

        params[parameter_key] = scaled_intercepts


def update_current_parameter_dictionary(self, list_position):
    """
    This function updates the module level dictionary self.current_parameters to contain a set of parameters
    relevant to a time period during simulation burn-in. LIST parameters[0] represent values for 2010-2014 and LIST
    parameters[1] represent values from 2015 onwards
    :param list_position: [0] 2010- 2014, [1] 2015 onwards
    """

    for key, value in self.parameters.items():
        if type(value) is list:
            self.current_parameters[key] = self.parameters[key][list_position]
        else:
            self.current_parameters[key] = self.parameters[key]


def store_dalys_in_mni(self, individual_id, mni_variable):
    """
    This function is called across the maternal health modules and stores onset/resolution dates for complications
    in an indiviudal's MNI dictionary to be used in report_daly_values
    :param individual_id: individual_id
    :param mni_variable: key of mni dict being assigned
    """
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Women no longer in the mni dict cannot accrue disability
    if individual_id not in mni:
        return

    mni[individual_id][mni_variable] = self.sim.date


def dummy_store_dalys_in_mni(individual_id, mni, mni_variable, date):
    """
    This function is called across the maternal health modules and stores onset/resolution dates for complications
    in an indiviudal's MNI dictionary to be used in report_daly_values
    :param individual_id: individual_id
    :param mni_variable: key of mni dict being assigned
    """

    # Women no longer in the mni dict cannot accrue disability
    if individual_id not in mni:
        return

    mni[individual_id][mni_variable] = date


def check_for_risk_of_death_from_cause(self, target, individual_id):
    """
    This function calculates the risk of death associated with one or more causes being experience by an individual and
    determines if they will die and which of a number of competing cause is the primary cause of death
    :param target: 'mother' or 'neonate
    :param individual_id: individual_id of woman at risk of deaht
    return: cause of death or False
    """
    params = self.current_parameters
    df = self.sim.population.props
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
    nci = self.sim.modules['NewbornOutcomes'].newborn_care_info

    causes = list()

    # todo: causes need to match

    if target == 'mother':
        mother = df.loc[individual_id]

        if (mother.ps_htn_disorders == 'severe_pre_eclamp' and mni[individual_id]['new_onset_spe']) or \
          (mother.pn_htn_disorders == 'severe_pre_eclamp' and mni[individual_id]['new_onset_spe']):
            causes.append('severe_pre_eclampsia')

        if mother.ps_htn_disorders == 'eclampsia' or mother.pn_htn_disorders == 'eclampsia':
            causes.append('eclampsia')

        if mother.ps_antepartum_haemorrhage != 'none' or mother.la_antepartum_haem != 'none':
            causes.append('antepartum_haemorrhage')

        if mother.ps_chorioamnionitis:
            causes.append('antenatal_sepsis')

        if mother.la_uterine_rupture:
            causes.append('uterine_rupture')

        if mother.la_sepsis or (mother.ps_chorioamnionitis and mother.ac_admitted_for_immediate_delivery != 'none'):
            # todo: ???? above
            causes.append('intrapartum_sepsis')

        if mother.la_sepsis_pp or mother.pn_sepsis_late_postpartum:
            causes.append('postpartum_sepsis')

        if mother.la_postpartum_haem:
            causes.append('postpartum_haemorrhage')

        if mother.pn_postpartum_haem_secondary:
            causes.append('secondary_postpartum_haemorrhage')

    elif target == 'neonate':
        child = df.loc[individual_id]

        if child.nb_early_onset_neonatal_sepsis or child.pn_sepsis_early_neonatal:
            causes.append('early_onset_neonatal_sepsis')

        if child.pn_sepsis_late_neonatal:
            causes.append('late_onset_neonatal_sepsis')

        if not nci[individual_id]['passed_through_week_one']:

            if (child.nb_encephalopathy == 'mild_enceph' or
                child.nb_encephalopathy == 'moderate_enceph' or
               child.nb_encephalopathy == 'severe_enceph'):
                causes.append('encephalopathy')

            if (child.nb_not_breathing_at_birth and
                (child.nb_encephalopathy == 'none') and
               (not child.nb_preterm_respiratory_distress)):
                causes.append('neonatal_respiratory_depression')

            if child.nb_early_preterm or child.nb_late_preterm:
                causes.append('preterm_other')

            if child.nb_preterm_respiratory_distress:
                causes.append('respiratory_distress_syndrome')

            if self.congeintal_anomalies.has_all(individual_id, 'heart'):
                causes.append('congenital_heart_anomaly')
            if self.congeintal_anomalies.has_all(individual_id, 'limb_musc_skeletal'):
                causes.append('limb_or_musculoskeletal_anomaly')
            if self.congeintal_anomalies.has_all(individual_id, 'urogenital'):
                causes.append('urogenital_anomaly')
            if self.congeintal_anomalies.has_all(individual_id, 'digestive'):
                causes.append('digestive_anomaly')
            if self.congeintal_anomalies.has_all(individual_id, 'other'):
                causes.append('other_anomaly')

    if causes:
        risks = dict()
        for cause in causes:
            if self == self.sim.modules['PregnancySupervisor']:
                risk = {cause: params[f'prob_{cause}_death']}

                risks.update(risk)

            elif self == self.sim.modules['Labour']:
                if cause == 'secondary_postpartum_haemorrhage':
                    risk = {cause: self.la_linear_models['postpartum_haemorrhage_death'].predict(
                        df.loc[[individual_id]],
                        received_blood_transfusion=mni[individual_id]['received_blood_transfusion'], )[individual_id]}
                else:
                    risk = {cause: self.la_linear_models[f'{cause}_death'].predict(
                        df.loc[[individual_id]],
                        received_blood_transfusion=mni[individual_id]['received_blood_transfusion'],
                        mode_of_delivery=mni[individual_id]['mode_of_delivery'],
                        chorio_in_preg=mni[individual_id]['chorio_in_preg'])[individual_id]}

                risks.update(risk)

            elif self == self.sim.modules['PostnatalSupervisor']:
                risk = {cause: params[f'cfr_{cause}']}
                if (cause == 'secondary_postpartum_haemorrhage') and \
                    (df.at[individual_id, 'pn_anaemia_following_pregnancy'] != 'none'):

                    risk[cause] = risk[cause] * params['rr_death_from_pph_with_anaemia']

                risks.update(risk)

            elif self == self.sim.modules['NewbornOutcomes']:

                if f'{cause}_death' in self.nb_linear_models.keys():
                    risk = {cause: self.nb_linear_models[f'{cause}_death'].predict(
                        df.loc[[individual_id]])[individual_id]}
                else:
                    risk = {cause: params[f'cfr_{cause}']}

                risks.update(risk)

        # Calculate the total risk of death for all causes that are present
        # result = 1.0 - math.prod(1.0 - [_cause for _cause in causes])
        result = 1.0
        for cause in risks:
            result *= (1.0 - risks[cause])

        # If random draw is less that the total risk of death, she will die and the primary cause is then
        # determined
        if self.rng.random_sample() < (1.0 - result):
            denominator = sum(risks.values())
            probs = list()

            # Cycle over each cause in the dictionary and divide CFR by the sum of the probabilities
            for cause in risks:
                risks[cause] = risks[cause] / denominator
                probs.append(risks[cause])

            # Now use the list of probabilities to conduct a weighted random draw to determine primary cause of death
            cause_of_death = self.rng.choice(causes, p=probs)

            return cause_of_death

        else:
            return False


def update_mni_dictionary(self, individual_id):
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = self.sim.population.props

    assert df.at[individual_id, 'is_pregnant']

    if self == self.sim.modules['PregnancySupervisor']:

        mni[individual_id] = {'delete_mni': False,  # if True, mni deleted in report_daly_values function
                              'abortion_onset': pd.NaT,
                              'abortion_haem_onset': pd.NaT,
                              'abortion_sep_onset': pd.NaT,
                              'eclampsia_onset': pd.NaT,
                              'mild_mod_aph_onset': pd.NaT,
                              'severe_aph_onset': pd.NaT,
                              'chorio_onset': pd.NaT,
                              'chorio_in_preg': False,  # use in predictor in newborn linear models
                              'ectopic_onset': pd.NaT,
                              'ectopic_rupture_onset': pd.NaT,
                              'gest_diab_onset': pd.NaT,
                              'gest_diab_diagnosed_onset': pd.NaT,
                              'gest_diab_resolution': pd.NaT,
                              'mild_anaemia_onset': pd.NaT,
                              'mild_anaemia_resolution': pd.NaT,
                              'moderate_anaemia_onset': pd.NaT,
                              'moderate_anaemia_resolution': pd.NaT,
                              'severe_anaemia_onset': pd.NaT,
                              'severe_anaemia_resolution': pd.NaT,
                              'mild_anaemia_pp_onset': pd.NaT,
                              'mild_anaemia_pp_resolution': pd.NaT,
                              'moderate_anaemia_pp_onset': pd.NaT,
                              'moderate_anaemia_pp_resolution': pd.NaT,
                              'severe_anaemia_pp_onset': pd.NaT,
                              'severe_anaemia_pp_resolution': pd.NaT,
                              'hypertension_onset': pd.NaT,
                              'hypertension_resolution': pd.NaT,
                              'obstructed_labour_onset': pd.NaT,
                              'sepsis_onset': pd.NaT,
                              'uterine_rupture_onset': pd.NaT,
                              'mild_mod_pph_onset': pd.NaT,
                              'severe_pph_onset': pd.NaT,
                              'secondary_pph_onset': pd.NaT,
                              'vesicovaginal_fistula_onset': pd.NaT,
                              'vesicovaginal_fistula_resolution': pd.NaT,
                              'rectovaginal_fistula_onset': pd.NaT,
                              'rectovaginal_fistula_resolution': pd.NaT,
                              'test_run': False,  # used by labour module when running some model tests
                              'pred_syph_infect': pd.NaT,  # date syphilis is predicted to onset
                              'new_onset_spe': False,
                              'cs_indication': 'none'
                              }

    elif self == self.sim.modules['Labour']:
        labour_variables = {'labour_state': None,
                            # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or Post Term (POTL)
                            'birth_weight': 'normal_birth_weight',
                            'birth_size': 'average_for_gestational_age',
                            'delivery_setting': None,  # home_birth, health_centre, hospital
                            'twins': df.at[individual_id, 'ps_multiple_pregnancy'],
                            'twin_count': 0,
                            'twin_one_comps': False,
                            'pnc_twin_one': 'none',
                            'bf_status_twin_one': 'none',
                            'eibf_status_twin_one': False,
                            'an_placental_abruption': df.at[individual_id, 'ps_placental_abruption'],
                            'corticosteroids_given': False,
                            'clean_birth_practices': False,
                            'abx_for_prom_given': False,
                            'abx_for_pprom_given': False,
                            'endo_pp': False,
                            'retained_placenta': False,
                            'uterine_atony': False,
                            'amtsl_given': False,
                            'cpd': False,
                            'mode_of_delivery': 'vaginal_delivery',
                            # vaginal_delivery, instrumental, caesarean_section
                            'hsi_cant_run': False,  # True (T) or False (F)
                            'sought_care_for_complication': False,  # True (T) or False (F)
                            'sought_care_labour_phase': 'none',
                            'referred_for_cs': False,  # True (T) or False (F)
                            'referred_for_blood': False,  # True (T) or False (F)
                            'received_blood_transfusion': False,  # True (T) or False (F)
                            'referred_for_surgery': False,  # True (T) or False (F)'
                            'death_in_labour': False,  # True (T) or False (F)
                            'cause_of_death_in_labour': [],
                            'single_twin_still_birth': False,  # True (T) or False (F)
                            'will_receive_pnc': 'none',
                            'passed_through_week_one': False
                            }

        mni[individual_id].update(labour_variables)

# TODO: further on birth as one function living here?
