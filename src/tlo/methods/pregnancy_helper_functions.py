"""
This file stores any functions that are called from multiple different modules in the Maternal and Perinatal Health
module suite
"""

import numpy as np
import pandas as pd


def get_list_of_items(self, item_list):
    """
    Uses get_item_code_from_item_name to return item codes for a list of named items
    :param self: module
    :param item_list: items for code look up
    """
    item_code_function = self.sim.modules['HealthSystem'].get_item_code_from_item_name
    codes = [item_code_function(item) for item in item_list]

    return codes


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

    if parameter_key not in ('prob_spontaneous_abortion_per_month', 'baseline_prob_early_labour_onset'):

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
        if isinstance(value, list):
            self.current_parameters[key] = self.parameters[key][list_position]
        else:
            if list_position == 0:
                self.current_parameters[key] = self.parameters[key]


def store_dalys_in_mni(individual_id, mni, mni_variable, date):
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


def check_if_delayed_careseeking(self, individual_id):
    """
    This function checks if a woman who is seeking care for treatment of a pregnancy/postnatal related emergency will
    experience either a type 1 or type 2 delay
    :param self: module
    :param individual_id: individual_id
    """
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

    if individual_id not in mni:
        return

    if self.rng.random_sample() < self.sim.modules['Labour'].current_parameters['prob_delay_one_two_fd']:
        mni[individual_id]['delay_one_two'] = True


def check_if_delayed_care_delivery(self, squeeze_factor, individual_id, hsi_type):
    """
    This function checks if a woman who is receiving care during a HSI will experience a type three delay to her care
    due to high squeeze
    :param self: module
    :param squeeze_factor: squeeze factor of HSI event
    :param individual_id: individual_id
    :param hsi_type: STR (bemonc, cemonc, an, pn)
    :return:
    """

    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

    if squeeze_factor > self.current_parameters[f'squeeze_threshold_for_delay_three_{hsi_type}']:
        mni[individual_id]['delay_three'] = True


def get_treatment_effect(delay_one_two, delay_three, treatment_effect, params):
    """
    Returns a requested treatment effect which may be modified if care was delayed
    :param delay_one_two: BOOL, if delay 1/2 has occurred
    :param delay_three: BOOL, if a delay 3 has occurred
    :param treatment_effect: STR, parameter housing the treatment effect
    :param params: module parameters
    :return: Treatment effect to be used in the linear model
    """

    # If they have experienced all delays, treatment effectiveness is reduced by greater amount
    if delay_one_two and delay_three:
        treatment_effect = 1 - ((1 - params[treatment_effect]) * params['treatment_effect_modifier_all_delays'])

    # Otherwise, if only one type of delay is experience the treatment effect is reduced by a lesser amount
    elif delay_one_two or delay_three:
        treatment_effect = 1 - ((1 - params[treatment_effect]) * params['treatment_effect_modifier_one_delay'])

    # If no delays occurred, maximum treatment effectiveness is applied
    else:
        treatment_effect = params[treatment_effect]

    return treatment_effect


def calculate_risk_of_death_from_causes(self, risks):
    """
    This function calculates risk of death in the context of one or more 'death causing' complications in a mother of a
    newborn. In addition it determines if the complication(s) will cause death or not. If death occurs the function
    returns the primary cause of death (or False)
    return: cause of death or False
    """

    result = 1.0
    for cause in risks:
        result *= (1.0 - risks[cause])

    # result = 1.0 - math.prod(1.0 - [_cause for _cause in causes])

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
        cause_of_death = self.rng.choice(list(risks.keys()), p=probs)

        # Return the primary cause of death so that it can be passed to the demography function
        return cause_of_death
    else:
        # Return false if death will not occur
        return False


def check_for_risk_of_death_from_cause_maternal(self, individual_id):
    """
    This function calculates the risk of death associated with one or more causes being experience by an individual and
    determines if they will die and which of a number of competing cause is the primary cause of death
    :param individual_id: individual_id of woman at risk of death
    return: cause of death or False
    """
    params = self.current_parameters
    df = self.sim.population.props
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

    causes = list()

    mother = df.loc[individual_id]

    # Cycle through mothers properties to ascertain what she is at risk of death from and store in a list

    if (mother.ps_htn_disorders == 'severe_pre_eclamp' and mni[individual_id]['new_onset_spe']) or \
       (mother.pn_htn_disorders == 'severe_pre_eclamp' and mni[individual_id]['new_onset_spe']):
        causes.append('severe_pre_eclampsia')

    if mother.ps_htn_disorders == 'eclampsia' or mother.pn_htn_disorders == 'eclampsia':
        causes.append('eclampsia')

    if (((mother.ps_antepartum_haemorrhage != 'none') or
         (mother.la_antepartum_haem != 'none')) and (self != self.sim.modules['PostnatalSupervisor'])):
        causes.append('antepartum_haemorrhage')

    if mother.ps_chorioamnionitis and (self == self.sim.modules['PregnancySupervisor']):
        causes.append('antenatal_sepsis')

    if mother.la_uterine_rupture:
        causes.append('uterine_rupture')

    if mother.la_sepsis or ((self == self.sim.modules['Labour']) and
                            mother.ps_chorioamnionitis and mother.ac_admitted_for_immediate_delivery != 'none'):
        causes.append('intrapartum_sepsis')

    if mother.la_sepsis_pp or mother.pn_sepsis_late_postpartum:
        causes.append('postpartum_sepsis')

    if mother.la_postpartum_haem:
        causes.append('postpartum_haemorrhage')

    if mother.pn_postpartum_haem_secondary:
        causes.append('secondary_postpartum_haemorrhage')

    # If this list is not empty, use either CFR parameters or linear models to calculate risk of death from each
    # complication she is experiencing and store in a dictionary, using each cause as the key
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
                        received_blood_transfusion=mni[individual_id]['received_blood_transfusion'],
                        delay_one_two=mni[individual_id]['delay_one_two'],
                        delay_three=mni[individual_id]['delay_three']
                    )[individual_id]}
                else:
                    risk = {cause: self.la_linear_models[f'{cause}_death'].predict(
                        df.loc[[individual_id]],
                        received_blood_transfusion=mni[individual_id]['received_blood_transfusion'],
                        mode_of_delivery=mni[individual_id]['mode_of_delivery'],
                        chorio_in_preg=mni[individual_id]['chorio_in_preg'],
                        delay_one_two=mni[individual_id]['delay_one_two'],
                        delay_three=mni[individual_id]['delay_three'])[individual_id]}
                risks.update(risk)

            elif self == self.sim.modules['PostnatalSupervisor']:
                risk = {cause: params[f'cfr_{cause}']}
                if (cause == 'secondary_postpartum_haemorrhage') and \
                   (df.at[individual_id, 'pn_anaemia_following_pregnancy'] != 'none'):

                    risk[cause] = risk[cause] * params['rr_death_from_pph_with_anaemia']
                risks.update(risk)

        # Call return the result from calculate_risk_of_death_from_causes function
        return calculate_risk_of_death_from_causes(self, risks)

    # if she is not at risk of death as she has no complications we return false to the module
    return False


def check_for_risk_of_death_from_cause_neonatal(self, individual_id):
    """
    This function calculates the risk of death associated with one or more causes being experience by an individual and
    determines if they will die and which of a number of competing cause is the primary cause of death
    :param individual_id: individual_id of woman at risk of death
    return: cause of death or False
    """
    params = self.current_parameters
    df = self.sim.population.props
    nci = self.sim.modules['NewbornOutcomes'].newborn_care_info

    causes = list()

    child = df.loc[individual_id]

    # Cycle through Newborns properties to ascertain what she is at risk of death from and store in a list
    if child.nb_early_onset_neonatal_sepsis or child.pn_sepsis_early_neonatal:
        causes.append('early_onset_sepsis')

    if child.pn_sepsis_late_neonatal:
        causes.append('late_onset_sepsis')

    # Risk of death for some complications is applied once, only in those who have yet to move to the postnatal module
    if not nci[individual_id]['passed_through_week_one']:

        if child.nb_encephalopathy != 'none':
            causes.append('encephalopathy')

        if ((child.nb_not_breathing_at_birth and
             (child.nb_encephalopathy == 'none') and
             not child.nb_preterm_respiratory_distress)):
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

    # If this list is not empty, use either CFR parameters or linear models to calculate risk of death from each
    # complication they experiencing and store in a dictionary, using each cause as the key
    if causes:
        risks = dict()
        for cause in causes:
            if f'{cause}_death' in self.nb_linear_models.keys():
                risk = {cause: self.nb_linear_models[f'{cause}_death'].predict(
                    df.loc[[individual_id]], delay=nci[individual_id]['third_delay'])[individual_id]}
            else:
                risk = {cause: params[f'cfr_{cause}']}

            risks.update(risk)

        # Return the result from calculate_risk_of_death_from_causes function (returns primary cause of death or False)
        return calculate_risk_of_death_from_causes(self, risks)

    # if they is not at risk of death as they has no complications we return False to the module
    return False


def update_mni_dictionary(self, individual_id):
    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = self.sim.population.props

    if self == self.sim.modules['PregnancySupervisor']:

        mni[individual_id] = {'delay_one_two': False,
                              'delay_three': False,
                              'delete_mni': False,  # if True, mni deleted in report_daly_values function
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
