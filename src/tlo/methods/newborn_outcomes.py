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
logger.setLevel(logging.INFO)


class NewbornOutcomes(Module):
    """This module is responsible for the outcomes of newborns immediately following delivery and interventions provided
     by skilled birth attendants to newborns following delivery"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath
        self.newborn_care_info = dict()

    PARAMETERS = {
        'base_incidence_low_birth_weight': Parameter(
            Types.REAL, 'baseline incidence of low birth weight for neonates'),
        'mean_birth_weights': Parameter(
            Types.LIST, 'list of mean birth weights from gestational age at birth 24-41 weeks'),
        'prob_congenital_ba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_cba_type': Parameter(
            Types.LIST, 'Probability of types of CBA'),
        'odds_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline odds of a neonate developing sepsis following birth (early onset)'),
        'odds_ratio_sepsis_partiy0': Parameter(
            Types.REAL, 'odds ratio for developing neonatal sepsis if this is the mothers first child'),
        'odds_ratio_sepsis_preterm': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for preterm neonates'),
        'odds_ratio_sepsis_lbw': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for low birth weight neonates'),
        'odds_ratio_sepsis_vlbw': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for very low birth weight neonates'),
        'prob_failure_to_transition': Parameter(
            Types.REAL, 'baseline probability of a neonate developing intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'odds_encephalopathy': Parameter(
            Types.REAL, 'baseline odds of a neonate developing encephalopathy of any severity following birth'),
        'odds_enceph_neonatal_sepsis': Parameter(
            Types.REAL, 'odds ratio of neonatal encephalopathy if the neonate is also septic'),
        'odds_enceph_males': Parameter(
            Types.REAL, 'odds ratio for encephalopathy if the neonate is male'),
        'odds_enceph_obstructed_labour': Parameter(
            Types.REAL, 'odds ratio for encephalopathy if the mothers labour was obstructed'),
        'odds_enceph_hypertension': Parameter(
            Types.REAL, 'odds ratio for encephalopathy if the mothers suffered from hypertension in pregnancy'),
        'odds_enceph_acute_event': Parameter(
            Types.REAL, 'odds ratio for encephalopathy if the mothers experience an acute event in labour'),
        'prob_enceph_severity': Parameter(
            Types.LIST, 'probability of the severity of encephalopathy in a newborn who is encephalopathic'),
        'prob_retinopathy_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing retinopathy of prematurity '),
        'prob_retinopathy_severity': Parameter(
            Types.LIST,
            'probabilities of severity of retinopathy'),
        'prob_low_birth_weight': Parameter(
            Types.REAL, 'baseline probability of a neonate being born low birth weight'),
        'prob_early_breastfeeding_hf': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at a health facility'),
        'prob_early_breastfeeding_hb': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at home'),
        'prob_facility_offers_kmc': Parameter(
            Types.REAL, 'probability that the facility in which a low birth weight neonate is born will offer kangaroo'
                        ' mother care for low birth weight infants delivered at home'),
        'prob_successful_resuscitation': Parameter(
            Types.REAL, 'probability newborn resuscitation will be successful'),
        'rr_sepsis_tetracycline': Parameter(
            Types.REAL, 'relative risk of neonatal sepsis following tetracycline ointment treatment'),
        'rr_sepsis_cord_care': Parameter(
            Types.REAL, 'relative risk of neonatal sepsis following chlorhexadine cord care'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_encephalopathy': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to encephalopathy'),
        'cfr_failed_to_transition': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'cfr_preterm_birth': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'prob_care_seeking_for_complication': Parameter(
            Types.REAL, 'baseline probability that a mother will seek care for an unwell neonate following delivery'),
        'sensitivity_of_assessment_of_neonatal_sepsis_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of neonatal sepsis in level 1 health centre'),
        'sensitivity_of_assessment_of_neonatal_sepsis_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of neonatal sepsis in level 1 hospital'),
        'sensitivity_of_assessment_of_ftt_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of failure to transition in level 1 health centre'),
        'sensitivity_of_assessment_of_ftt_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of failure to transition in level 1 hospital'),
        'sensitivity_of_assessment_of_lbw_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of low birth weight in level 1 health centre'),
        'sensitivity_of_assessment_of_lbw_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment of low birth weight in level 1 hospital'),
    }

    PROPERTIES = {
        'nb_early_preterm': Property(Types.BOOL, 'whether this neonate has been born early preterm (24-33 weeks '
                                                 'gestation)'),
        'nb_late_preterm': Property(Types.BOOL, 'whether this neonate has been born late preterm (34-36 weeks '
                                                'gestation)'),
        'nb_congenital_anomaly': Property(Types.CATEGORICAL, 'Congenital Anomalies: None, Orthopedic, Gastrointestinal,'
                                                             'Neurological, Cosmetic, Other',
                                          #  todo: May need more specificity
                                          categories=['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whether his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_treatment_for_neonatal_sepsis': Property(Types.CATEGORICAL, 'If this neonate has received treatment for '
                                                                        'neonatal sepsis, and how promptly',
                                                     categories=['none', 'prompt_treatment', 'delayed_treatment']),
        'nb_failed_to_transition': Property(Types.BOOL, 'whether this neonate has failed to transition to breathing on '
                                                        'their own following birth'),
        'nb_received_neonatal_resus': Property(Types.CATEGORICAL, 'If this neonate has received treatment for '
                                                                        'neonatal sepsis, and how promptly',
                                                     categories=['none', 'prompt_treatment', 'delayed_treatment']),
        'nb_encephalopathy': Property(Types.CATEGORICAL, 'None, mild encephalopathy, moderate encephalopathy, '
                                                         'severe encephalopathy',
                                      categories=['none', 'mild_enceph', 'moderate_enceph', 'severe_enceph']),
        'nb_retinopathy_prem': Property(Types.CATEGORICAL, 'Level of visual disturbance due to retinopathy of'
                                                           ' prematurity: None, mild, moderate, severe, blindness',
                                        categories=['none', 'mild', 'moderate', 'severe', 'blindness']),
        'nb_ongoing_impairment': Property(Types.CATEGORICAL, 'none, mild motor, mild motor and cognitive, '
                                                             'moderate motor, moderate motor and cognitive, '
                                                             'severe motor, severe motor and cognitive',
                                          categories=['none', 'mild_mot', 'mild_mot_cog', 'mod_mot', 'mod_mot_cog',
                                                      'severe_mot', ' severe_mot_cog']),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'extremely low birth weight (<1000g), '
                                                                  ' very low birth weight (<1500g), '
                                                                  'low birth weight (<2500g),'
                                                                  'normal birth weight (>2500g)',
                                    categories=['extremely_low_birth_weight', 'very_low_birth_weight', 
                                                'low_birth_weight', 'normal_birth_weight']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL,'size for gestatiaonal age catagories',
                                                categories=['small_for_gestational_age', 'average_for_gestational_age',
                                                            'large_for_gestational_age']),
        'nb_early_breastfeeding': Property(Types.BOOL, 'whether this neonate is exclusively breastfed after birth'),
        'nb_kangaroo_mother_care': Property(Types.BOOL, 'whether this neonate received kangaroo mother care following '
                                                        'birth'),
        'nb_death_after_birth': Property(Types.BOOL, 'whether this child has died following complications after birth'),
        'nb_death_after_birth_date': Property(Types.DATE, 'date on which the child died after birth'),
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_NewbornOutcomes.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # TODO: Discuss with team the best way to organise and apply DALY weights for newborns

        # The below code is commented out whilst we determine the best way too apply DALYs and prelonged disability to
        # newborns following birth...

#        if 'HealthBurden' in self.sim.modules.keys():
#            params['nb_daly_wts'] = {
#                'mild_motor_cognitive_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(357),
#                'mild_motor_cognitive_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(359),
#                'mild_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(371),
#                'moderate_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(378),
#                'severe_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(383),
#                'mild_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(372),
#                'moderate_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(377),
#                'severe_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(375),
#                'mild_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(373),
#                'moderate_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(379),
#                'severe_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(366),
#                'mild_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(404),
#                'moderate_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(405),
#                'severe_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(402),
#                'blindness_rptb': self.sim.modules['HealthBurden'].get_daly_weight(386),
#                'mild_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(416),
#                'moderate_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(411),
#                'severe_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(410),
#                'mild_motor_cognitive_enceph': self.sim.modules['HealthBurden'].get_daly_weight(419),
#                'severe_motor_cognitive_enceph': self.sim.modules['HealthBurden'].get_daly_weight(420),
#                'mild_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(431),
#                'moderate_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(438),
#                'severe_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(435),
#                'severe_infection_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(436),
#                'mild_motor_cognitive_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(441),
#                'mild_motor_cognitive_haemolytic': self.sim.modules['HealthBurden'].get_daly_weight(457),
#                'severe_motor_cognitive_haemolytic': self.sim.modules['HealthBurden'].get_daly_weight(455)}

# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # All linear equations used in this module are stored within the nb_newborn_equations parameter below

        params = self.parameters
        #  TODO: treatment effects in death equations
        params['nb_newborn_equations'] = {

            'neonatal_sepsis_home_birth': LinearModel(
             LinearModelType.LOGISTIC,
             params['odds_early_onset_neonatal_sepsis'],
             Predictor('la_parity').when('0', params['odds_ratio_sepsis_partiy0']),
             Predictor('nb_early_preterm').when(True, params['odds_ratio_sepsis_preterm']),
             Predictor('nb_late_preterm').when(True, params['odds_ratio_sepsis_preterm']),
             Predictor('nb_low_birth_weight_status').when('low_birth_weight', params['odds_ratio_sepsis_lbw']).when(
                 'very_low_birth_weight', params['odds_ratio_sepsis_vlbw']).when('extremely_low_birth_weight', params[
                    'odds_ratio_sepsis_vlbw'])),

            'neonatal_sepsis_facility_delivery': LinearModel(
             LinearModelType.LOGISTIC,  # todo: this needs to be nci[child_id]['ongoing_sepsis_risk']
             params['odds_early_onset_neonatal_sepsis'],
             Predictor('la_parity').when('0', params['odds_ratio_sepsis_partiy0']),
             Predictor('nb_early_preterm').when(True, params['odds_ratio_sepsis_preterm']),
             Predictor('nb_late_preterm').when(True, params['odds_ratio_sepsis_preterm']),
             Predictor('nb_low_birth_weight_status').when('low_birth_weight', params['odds_ratio_sepsis_lbw']).when(
                    'very_low_birth_weight', params['odds_ratio_sepsis_vlbw']).when('extremely_low_birth_weight',
                                                                                    params[
                                                                                        'odds_ratio_sepsis_vlbw'])),

            'neonatal_sepsis_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_neonatal_sepsis'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'encephalopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['odds_encephalopathy'],
             Predictor('nb_early_onset_neonatal_sepsis').when(True,  params['odds_enceph_neonatal_sepsis']),
             Predictor('sex').when('M',  params['odds_enceph_males'])),
            # todo: use below as external variables?
            # Predictor('la_obstructed_labour').when(True, params['odds_enceph_obstructed_labour']),
            # Predictor('ps_gestational_htn').when(True, params['odds_enceph_hypertension']),
            # Predictor('ps_mild_pre_eclamp').when(True, params['odds_enceph_hypertension']),
            # Predictor('ps_severe_pre_eclamp').when(True, params['odds_enceph_hypertension'])),

            'encephalopathy_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_encephalopathy'],
             Predictor('nb_encephalopathy').when('mild', 2).when('moderate', 3).when('severe', 4)),

            'failure_to_transition': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_failure_to_transition'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'failed_to_transition_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_failed_to_transition'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'retinopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_retinopathy_preterm'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'preterm_birth_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_preterm_birth'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'care_seeking_for_complication': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_care_seeking_for_complication'],
             Predictor('age_years').when('.between(0,2)', 1))

        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_congenital_anomaly'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_treatment_for_neonatal_sepsis'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_failed_to_transition'] = False
        df.loc[df.is_alive, 'nb_received_neonatal_resus'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_encephalopathy'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_retinopathy_prem'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_ongoing_impairment'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'].values[:] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'].values[:] = 'average_for_gestational_age'
        df.loc[df.is_alive, 'nb_early_breastfeeding'] = False
        df.loc[df.is_alive, 'nb_kangaroo_mother_care'] = False
        df.loc[df.is_alive, 'nb_death_after_birth'] = False
        df.loc[df.is_alive, 'nb_death_after_birth_date'] = pd.NaT

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        # We define the diagnostic tests that will be called within the health system interactions
        # Neonatal sepsis...
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_neonatal_sepsis_hc=DxTest(
                property='nb_early_onset_neonatal_sepsis',
                sensitivity=self.parameters['sensitivity_of_assessment_of_neonatal_sepsis_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_neonatal_sepsis_hp=DxTest(
                property='nb_early_onset_neonatal_sepsis',
                sensitivity=self.parameters['sensitivity_of_assessment_of_neonatal_sepsis_hp'], ))

        # Failure to transition...
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_failure_to_transition_hc=DxTest(
                property='nb_failed_to_transition',
                sensitivity=self.parameters['sensitivity_of_assessment_of_ftt_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_failure_to_transition_hp=DxTest(
                property='nb_failed_to_transition',
                sensitivity=self.parameters['sensitivity_of_assessment_of_ftt_hp'], ))

        # Low birth weight...
        # todo: issue with this not working for categorical properties
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_low_birth_weight_hc=DxTest(
                property='nb_low_birth_weight_status',
                sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_low_birth_weight_hp=DxTest(
                property='nb_low_birth_weight_status',
                sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hp'], ))


    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        return self.rng.random_sample(size=1) < eq.predict(self.sim.population.props.loc[[person_id]])[person_id]

    def set_birth_weight(self, child_id, gestation_at_birth):
        """This function generates a birth weight for each newborn, drawn randomly from a normal distribution around a
        mean birthweight for each gestational age in weeks. Babies below the 10th percentile for their gestational age
        are small for gestational age"""

        params = self.parameters
        df = self.sim.population.props

        # We dont have data for mean birth weight after 41 weeks, so currently cap at 41
        if gestation_at_birth > 41:
            gestation_at_birth = 41

        # Mean birth weights for each gestational age are listed in a parameter starting at 24 weeks
        # We select the correct mean birth weight from the parameter
        mean_birth_weight_list_location = gestation_at_birth - 24

        # We randomly draw this newborns weight from a normal distribution around the mean for their gestation
        birth_weight = np.random.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=20)
        birth_weight_distribution = np.random.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location],
                                                     scale=20, size=10000) 
        # todo: size needs to be yearly births of the specific gestation?

        # Then we calculate the 10th and 90th percentile, the cutoffs for SGA and LGA
        small_for_gestational_age_cutoff = np.percentile(birth_weight_distribution, 10)
        large_for_gestational_age_cutoff = np.percentile(birth_weight, 90)

        # Make the appropriate changes to the data frame and log accordingly
        if birth_weight > 2500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        elif 1500 < birth_weight < 2500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'low_birth_weight'
        elif birth_weight < 1500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'very_low_birth_weight'
        elif birth_weight < 1000:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'extremely_low_birth_weight'

        if birth_weight < 2500:  # todo: log the VLBW or ELBW separately
            logger.info('%s|low_birth_weight_newborn|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})
            
        if birth_weight < small_for_gestational_age_cutoff:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'small_for_gestational_age'
            logger.info('%s|small_for_gestational_age_newborn|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        elif birth_weight > large_for_gestational_age_cutoff:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'large_for_gestational_age'
        else:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'

    def set_neonatal_death_status(self, individual_id, cause):
        """This function  is called for neonates that have experienced a complication after birth and determines if it
        will cause their death. Properties in the DF are set accordingly."""
        df = self.sim.population.props
        params = self.parameters

        if self.eval(params['nb_newborn_equations'][f'{cause}_death'], individual_id):
            df.at[individual_id, 'nb_death_after_birth'] = True
            df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

            logger.debug(F'This is NewbornOutcomes scheduling a death for person %d on date %s who died due to {cause}'
                         'complications following birth', individual_id, self.sim.date)

    def on_birth(self, mother_id, child_id):
        """The on_birth function of this module is used to apply the probability that a newborn will experience
        complications following delivery (which may or may not be attributable to the delivery process). This section of
        code is subject to change as it contains some dummy code (low birth weight/'small_for_gestational_age'
        application) and needs review by a clinician"""

        df = self.sim.population.props
        params = self.parameters
        nci = self.newborn_care_info

        # mni dictionary is deleted on maternal death/disease reset. Condition on mother being alive to read in mni
        # from labour
        if df.at[mother_id, 'is_alive'] or (~df.at[mother_id, 'is_alive'] &
                                            ~df.at[mother_id, 'la_intrapartum_still_birth']):
            mni = self.sim.modules['Labour'].mother_and_newborn_info
            m = mni[mother_id]

        df.at[child_id, 'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_congenital_anomaly'] = 'none'
        df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = False
        df.at[child_id, 'nb_failed_to_transition'] = False
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_ongoing_impairment'] = 'none'
        df.at[child_id, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child_id, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.at[child_id, 'nb_early_breastfeeding'] = False
        df.at[child_id, 'nb_kangaroo_mother_care'] = False
        df.at[child_id, 'nb_death_after_birth'] = False
        df.at[child_id, 'nb_death_after_birth_date'] = pd.NaT

        child = df.loc[child_id]

        # Here we set the variables for newborns delivered at less than 37 weeks, allocating them to either late or
        # early  preterm based on the gestation at labour

        if m['labour_state'] == 'early_preterm_labour':
            df.at[child_id, 'nb_early_preterm'] = True
            logger.info('%s|early_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        elif m['labour_state'] == 'late_preterm_labour':
            df.at[child_id, 'nb_late_preterm'] = True
            logger.info('%s|late_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})
        else:
            df.at[child_id, 'nb_early_preterm'] = False
            df.at[child_id, 'nb_late_preterm'] = False

        # Check no children born at term or postterm women are incorrectly categorised as preterm
        if m['labour_state'] == 'TL':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']
        if m['labour_state'] == 'POTL':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']

        if child.nb_early_preterm or child.nb_late_preterm:
            logger.info('%s|preterm_birth|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        # TODO: will this log correctly considering intrapartum stillbirths will undergo instantaneous death
        # TODO: will labour initialise first ensuring InstantaneousDeath fires before this logic?

        # Here we apply the prevalence of congenital birth anomalies in infants who have survived to delivery
        if child.is_alive and ~m['stillbirth_in_labour']:
            if self.rng.random_sample() < params['prob_congenital_ba']:
                etiology = ['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']
                probabilities = params['prob_cba_type']
                random_choice = self.rng.choice(etiology, size=1, p=probabilities)
                df.at[child_id, 'nb_congenital_anomaly'] = random_choice

        # For all newborns we first generate a dictionary that will store the prophylactic interventions then receive at
        # birth if delivered in a facility

            # Check the mothers MNI dictionary has data
            assert mni[mother_id]['risk_newborn_sepsis'] is not None

            # Set up NCI dictionary
            nci[child_id] = {'cord_care': False,
                             'vit_k': False,
                             'tetra_eye_d': False,
                             'proph_abx': False,
                             'ongoing_sepsis_risk': mni[mother_id]['risk_newborn_sepsis'],
                             'delivery_attended':  mni[mother_id]['delivery_attended'],
                             'delivery_facility_type': mni[mother_id]['delivery_facility_type']}
            # check this is carrying over correctly

            # TODO:  review in context of properties

    # =================================== BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE ===================================

            # The set birth weight function is used to determine this newborns birth weight and size for gestational age
            self.set_birth_weight(child_id, df.at[mother_id, 'ps_gestational_age_in_weeks'])

    # ================================== COMPLICATIONS FOLLOWING BIRTH ================================================
            # Here, using linear equations, we determine individual newborn risk of complications following delivery.
            # This risk is either stored or applied depending on the complication, as detailed below

            # SEPSIS....
            if m['delivery_setting'] == 'home_birth' and self.eval(params['nb_newborn_equations'][
                                                                       'neonatal_sepsis_home_birth'], child_id):
                df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True

                logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            child_id, self.sim.date)
                logger.info('%s|early_onset_nb_sep_hb|%s', self.sim.date, {'person_id': child_id})

            elif m['delivery_setting'] == 'facility_delivery':
                nci[child_id]['ongoing_sepsis_risk'] = params['nb_newborn_equations'][
                    'neonatal_sepsis_facility_delivery'].predict(df.loc[[child_id]])[child_id]

            # ENCEPHALOPATHY....
            if ~child.nb_early_preterm & ~child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['encephalopathy'], child_id):
                    # For a newborn who is encephalopathic we then set the severity
                    # todo: should we have individual equations for each severity?

                    severity_enceph = self.rng.choice(('mild', 'moderate', 'severe'), p=params['prob_enceph_severity'])
                    if severity_enceph == 'mild':
                        df.at[child_id, 'nb_encephalopathy'] = 'mild_enceph'
                    elif severity_enceph == 'moderate':
                        df.at[child_id, 'nb_encephalopathy'] = 'moderate_enceph'
                    else:
                        df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'

                    # Check all encephalopathy cases receive a grade
                    assert df.at[child_id, 'nb_encephalopathy'] != 'none'
                    # todo: correct to only apply to term infants only?

            # PRETERM BIRTH COMPS
            # retinopathy is a specific complication of prematurity that we do apply explicitly to map with DALYs
            if (child.nb_early_preterm or child.nb_late_preterm) and self.eval(params['nb_newborn_equations'][
                                                                                   'retinopathy'], child_id):
                random_draw = self.rng.choice(('mild', 'moderate', 'severe', 'blindness'),
                                                      p=params['prob_retinopathy_severity'])
                df.at[child_id, 'nb_retinopathy_prem'] = random_draw
                logger.debug(f'Neonate %d has developed {random_draw }retinopathy of prematurity',
                             child_id)

            # FAILURE TO TRANSITION...
            if df.at[child_id, 'nb_encephalopathy'] != 'none':
                # (cally tan) All encephalopathic children will need some form of neonatal resuscitation
                df.at[child_id, 'nb_failed_to_transition'] = True

            elif self.eval(params['nb_newborn_equations']['failure_to_transition'], child_id):
                df.at[child_id, 'nb_failed_to_transition'] = True
                logger.debug(f'Neonate %d has failed to transition from foetal to neonatal state and is not breathing '
                             f'following delivery', child_id)

    # ===================================== DISABILITY/LONG TERM IMPAIRMENT  ==========================================

            # This is a placeholder for dealing with properties that will capture the long term outcome sequalae
            # associated with a number of these complications. This will likely be stored in the nb_ongoing_impairment
            # property

    # ======================================= SCHEDULING NEWBORN CARE  ================================================
        # Neonates who were delivered in a facility are automatically scheduled to receive care after birth
            if m['delivery_setting'] == 'facility_delivery':
                event = HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(self,
                                                                                                  person_id=child_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesSkilledAttendance'
                             'FollowingBirthFacilityLevel1 for child %d following a facility delivery', child_id)
                # TODO: currently level 2 event is not ever scheduled

    # ========================================== CARE SEEKING  ======================================================

            # If this neonate was delivered at home and develops a complications we determine the likelihood of care
            # seeking
            # TODO: care seeking just for preterm birth?
            if (m['delivery_setting'] == 'home_birth') & (child.nb_failed_to_transition or
                                                          child.nb_early_onset_neonatal_sepsis or
                                                          child.nb_encephalopathy != 'none'):
                if self.eval(params['nb_newborn_equations']['care_seeking_for_complication'], child_id):
                    event = HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(self,
                                                                                                      person_id=
                                                                                                      child_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                    logger.debug('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesSkilledAttendance'
                                 'FollowingBirthFacilityLevel1 for child %d whose mother has sought care after a '
                                 'complication has developed following a home_birth', child_id)

    # ============================================ BREAST FEEDING AT HOME ============================================
                # Using DHS data we apply a one of probability that women who deliver at home will initiate
                # breastfeeding within one hour of birth
            if (m['delivery_setting'] == 'home_birth') and (~child.nb_failed_to_transition and
                                                            ~child.nb_early_onset_neonatal_sepsis and
                                                            (child.nb_encephalopathy == 'none')):

                if self.rng.random_sample() < params['prob_early_breastfeeding_hb']:
                    df.at[child_id, 'nb_early_breastfeeding'] = True
                    logger.debug(
                        'Neonate %d has started breastfeeding within 1 hour of birth', child_id)
                else:
                    logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', child_id)

    # ===================================== SCHEDULING NEWBORN DEATH EVENT  ============================================
                # All newborns are then scheduled to pass through a newborn death event to determine likelihood of death
                # in the presence of complications

            self.sim.schedule_event(NewbornDeathEvent(self, child_id), self.sim.date + DateOffset(days=3))
            logger.info('This is NewbornOutcomesEvent scheduling NewbornDeathEvent for person %d', child_id)

            # In the rare instance that a baby has been delivered following a mothers death in labour, we now delete
            # the mni dictionary record for that woman
            if m['death_in_labour']:
                del mni[mother_id]

    def on_hsi_alert(self, person_id, treatment_id):

        logger.info('This is NewbornOutcomes, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        logger.debug('This is Newborn Outcomes reporting my health values')

        # This section of code is just a dummy whilst we review how best to apply DALYs in the newborn module

        df = self.sim.population.props

        health_values_1 = df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'].map(
                    {False: 0, True: 0.324})  # p['daly_wt_mild_motor_sepsis']
        health_values_1.name = 'Sepsis Motor Impairment'
        health_values_df = pd.concat([health_values_1.loc[df.is_alive]], axis=1)

        return health_values_df


class NewbornDeathEvent(Event, IndividualScopeEventMixin):
    """This is the NewbornDeathEvent. It is scheduled for all newborns via the on_birth function. This event determines
    if those newborns who have experienced any complications will die due to them. This event will likely be changed as
     at present we dont deal with death of newborns due to complications of pre-term birth or congenital anomalies
     awaiting discussion regarding ongoing newborn care (NICU)"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        child = df.loc[individual_id]

        # Check the correct amount of time has passed between birth and the death event
        assert (self.sim.date - df.at[individual_id, 'date_of_birth']) == pd.to_timedelta(3, unit='D')

        # Using the set_neonatal_death_status function, defined above, it is determined if newborns who have experienced
        # complications will die because of them. Successful treatment in a HSI will turn off theses properties meaning
        # newborns will not die due to the complication. This logic needs to be reviewed

        if child.nb_early_onset_neonatal_sepsis:
            self.module.set_neonatal_death_status(individual_id, cause='neonatal_sepsis')

        if child.nb_encephalopathy != 'none':
            self.module.set_neonatal_death_status(individual_id, cause='encephalopathy')

        if child.nb_failed_to_transition:
            self.module.set_neonatal_death_status(individual_id, cause='failed_to_transition')

        if child.nb_early_preterm or child.nb_early_preterm:
            self.module.set_neonatal_death_status(individual_id, cause='preterm_birth')

        child = df.loc[individual_id]
        if child.nb_death_after_birth:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)

        #  Todo: make sure this is delayed enough following HSI?
        #  TODO: Tim C suggested we need to create an offset (using a distribution?) so we're generating deaths for the
        #   first 48 hours
        # TODO: use append to add cause of death to mni?

# ================================ HEALTH SYSTEM INTERACTION EVENTS ================================================

class HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1. This event is scheduled by
    the on_birth function of this module, automatically for neonates who delivered in facility and via a care seeking
    equation for those delivered at home. This event manages initial care of newborns following birth at level 1
    facilities. This event also houses assessment and treatment of complications following delivery (sepsis and failure
    to transition). Eventually it will scheduled additional treatment for those who need it."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # ???

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        nci = self.module.newborn_care_info
        params = self.module.parameters
        df = self.sim.population.props

        logger.info('This is HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel: child %d is '
                    'receiving care following delivery in a health facility on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # This HSI follows a similar structure as Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1
        # This function manages the administration of essential newborn care and prophylaxsis
        def prophylactic_interventions():
            # The required consumables are defined
            item_code_tetracycline = pd.unique(consumables.loc[consumables['Items'] == 'Tetracycline eye ointment '
                                                                                       '1%_3.5_CMST', 'Item_Code'])[0]

            item_code_vit_k = pd.unique(consumables.loc[consumables['Items'] == 'vitamin K1  (phytomenadione) 1 mg/ml, '
                                                                                '1 ml, inj._100_IDA', 'Item_Code'])[0]
            item_code_vit_k_syringe = pd.unique(consumables.loc[consumables['Items'] == 'Syringe,  disposable 2ml,  '
                                                                                        'hypoluer with 23g needle_each_'
                                                                                        'CMST',
                                                                'Item_Code'])[0]
            consumables_newborn_care = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_tetracycline: 1, item_code_vit_k: 1, item_code_vit_k_syringe: 1}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_newborn_care, to_log=True)

            # CORD CARE
            # Consumables for cord care are recorded in the labour module, as part of the skilled birth attendance
            # package. This reduces the neonates risk of sepsis.
            nci[person_id]['ongoing_sepsis_risk'] = nci[person_id]['ongoing_sepsis_risk'] * params[
                'rr_sepsis_cord_care']

            # Tetracycline eye care and vitamin k prophylaxis are conditioned on the availability of consumables

            # TETRACYCLINE
            if outcome_of_request_for_consumables['Item_Code'][item_code_tetracycline]:
                logger.debug('Neonate %d has received tetracycline eye drops to reduce sepsis risk following a facility'
                             'delivery', person_id)
                nci[person_id]['ongoing_sepsis_risk'] = nci[person_id]['ongoing_sepsis_risk'] * params[
                    'rr_sepsis_tetracycline']
            else:
                logger.debug('This facility has no tetracycline and therefore was not given')

            # VITAMIN K
            if outcome_of_request_for_consumables['Item_Code'][item_code_vit_k] and \
               outcome_of_request_for_consumables['Item_Code'][item_code_vit_k_syringe]:
                logger.debug('Neonate %d has received vitamin k prophylaxis following a facility delivery', person_id)
                nci[person_id]['vit_k'] = True
                # TODO: whats the point of this
            else:
                logger.debug('This facility has no vitamin K and therefore was not given')

            # EARLY INITIATION OF BREAST FEEDING
            # A probably that early breastfeeding will initiated in a facility is applied
            if self.module.rng.random_sample() < params['prob_early_breastfeeding_hf']:
                df.at[person_id, 'nb_early_breastfeeding'] = True
                logger.debug('Neonate %d has started breastfeeding within 1 hour of birth', person_id)
            else:
                logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', person_id)
                # TODO: variation between HC and HP?

            # TODO: some consumables are counted within the delivery packages -i.e. chlorhex
            # TODO: Determine if neonates are given antibiotics for maternal risk factors

        # If this neonates delivery is attended, the function will run
        if nci[person_id]['delivery_attended']:
            prophylactic_interventions()
        else:
            # Otherwise they receives no benefit of prophylaxis
            logger.debug('neonate %d received no prophylaxis as they were delivered unattended', person_id)

        # --------------------------------- RECALCULATE SEPSIS RISK -------------------------------------

        # Following the administration of prophylaxis we determine if this neonate will develop sepsis
        if self.module.eval(params['nb_newborn_equations']['neonatal_sepsis_facility_delivery'], person_id):
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True

            logger.debug('Neonate %d has developed early onset sepsis following a facility delivery on date %s',
                         person_id, self.sim.date)
            logger.info('%s|early_onset_nb_sep_hc|%s', self.sim.date, {'person_id': person_id})

        # This function manages kangaroo mother care for low birth weight neonates
        def kangaroo_mother_care(facility_type):
            # We determine if staff will correctly identify if the neonate is low birth weight, and then initiate KMC
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_low_birth_weight_'
            f'{facility_type}', hsi_event=self):
                df.at[person_id, 'nb_kangaroo_mother_care'] = True
                logger.debug('Neonate %d has been correctly identified as being low birth weight, and kangaroo mother '
                             'care has been initiated', person_id, self.sim.date)
                # TODO: KMC reduces risk of sepsis at discharge. Maybe shouldn't be applied here?

        # Only stable neonates are assessed for KMC as per guidelines
        if nci[person_id]['delivery_attended'] and (~df.at[person_id, 'nb_early_onset_neonatal_sepsis'] and
                                                    ~df.at[person_id, 'nb_failed_to_transition'] and
                                                    df.at[person_id, 'nb_encephalopathy'] != 'none'):
            # Likelihood of correctassessmentt and treatment varied by facility type
            if nci[person_id]['delivery_facility_type'] == 'health_centre':
                kangaroo_mother_care('hc')
            elif nci[person_id]['delivery_facility_type'] == 'health_centre':
                kangaroo_mother_care('hp')

        # This function manages initiation of neonatal resuscitation
        def assessment_and_initiation_of_neonatal_resus(facility_type):
            # Required consumables are defined
            pkg_code_resus = pd.unique(consumables.loc[consumables[
                                                         'Intervention_Pkg'] == 'Neonatal resuscitation '
                                                                                '(institutional)',
                                                       'Intervention_Pkg_Code'])[0]

            consumables_needed = {'Intervention_Package_Code': {pkg_code_resus: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed)

            # We determine if staff will correctly identify this neonate will require resuscitation
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                           f'assess_failure_to_transition_'
                                                                           f'{facility_type}', hsi_event=self):

                # Then, if the consumables are available,resuscitation is started. We assume this is delayed in
                # deliveries that are not attended
                if outcome_of_request_for_consumables:
                    if nci[person_id]['delivery_attended']:
                        df.at[person_id, 'nb_received_neonatal_resus'] = 'prompt_treatment'
                    else:
                        df.at[person_id, 'nb_received_neonatal_resus'] = 'delayed_treatment'

        if nci[person_id]['delivery_facility_type'] == 'health_centre':
            assessment_and_initiation_of_neonatal_resus('hc')
        if nci[person_id]['delivery_facility_type'] == 'hospital':
            assessment_and_initiation_of_neonatal_resus('hp')

        # This function manages the assessment and treatment of neonatal sepsis, and follows the same structure as
        # resuscitation
        def assessment_and_treatment_newborn_sepsis(facility_type):
            pkg_code_sep = pd.unique(consumables.loc[consumables[
                                                         'Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                                     'Intervention_Pkg_Code'])[0]

            consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed)

            if outcome_of_request_for_consumables:
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_low_birth_weight_'
                   f'{facility_type}', hsi_event=self):
                    if nci[person_id]['delivery_attended']:
                        df.at[person_id, 'nb_treatment_for_neonatal_sepsis'] = 'prompt_treatment'
                    else:
                        df.at[person_id, 'nb_treatment_for_neonatal_sepsis'] = 'delayed_treatment'

        if nci[person_id]['delivery_facility_type'] == 'health_centre':
            assessment_and_treatment_newborn_sepsis('hc')
        if nci[person_id]['delivery_facility_type'] == 'hospital':
            assessment_and_treatment_newborn_sepsis('hp')

        # ------------------------------ (To go here- referral for further care) ---------------------------------------

class HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel2'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # ???

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        nci = self.module.newborn_care_info
        params = self.module.parameters
        df = self.sim.population.props

        logger.info('This is HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel2: child %d is '
                    'receiving care following delivery in a health facility on date %s', person_id, self.sim.date)

        if nci[person_id]['delivery_attended']:
            self.module.prophylactic_interventions()

        if self.module.eval(params['nb_newborn_equations']['neonatal_sepsis_facility_delivery'], person_id):
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True

            logger.info('Neonate %d has developed early onset sepsis following a facility delivery on date %s',
                        person_id, self.sim.date)
            logger.info('%s|early_onset_nb_sep_hc|%s', self.sim.date, {'person_id': person_id})

        self.module.kangaroo_mother_care('hp')
        self.module.assessment_and_initiation_of_neonatal_resus('hp')
        self.module.assessment_and_treatment_newborn_sepsis('hp')

        # ------------------------------ (To go here- referral for further care) ---------------------------------------


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is NewbornOutcomesLoggingEvent. Currently it produces a yearly output of neonatal and infant mortality.
    It is incomplete as both neonatal and infant mortality will be effected by deaths outside of this module """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        df = self.sim.population.props

        # Here we calculated the infant mortality rate, deaths in the first year of life per 1000 live births
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births_sum = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        cumm_deaths = len(df.index[df.nb_death_after_birth & (df.nb_death_after_birth_date > one_year_prior) &
                          (df.nb_death_after_birth_date < self.sim.date)])

        if cumm_deaths == 0:
            imr = 0
        else:
            imr = cumm_deaths / live_births_sum * 1000

        logger.info(f'The infant mortality ratio on date %s for this year is {imr} per 1000 live births',
                    self.sim.date)

        # Here we calculated the neonatal mortality rate, deaths in the 28 days of life per 1000 live births
        neonatal_deaths = len(df.index[~df.is_alive & ((df.nb_death_after_birth_date - df.date_of_birth)
                                       < pd.Timedelta(28, unit='D'))])

        if neonatal_deaths == 0:
            nmr = 0
        else:
            nmr = cumm_deaths / live_births_sum * 1000

        logger.info(f'The neonatal mortality ratio on date %s for this year is {nmr} per 1000 live births',
                    self.sim.date)
