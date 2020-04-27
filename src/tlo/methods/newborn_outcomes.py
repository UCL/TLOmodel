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

        # This dictionary will store information related to the neonates delivery
        self.newborn_care_info = dict()

        # This dictionary will track incidence of complications of following birth
        self.NewbornComplicationTracker = dict()

    PARAMETERS = {
        'base_incidence_low_birth_weight': Parameter(
            Types.REAL, 'baseline incidence of low birth weight for neonates'),
        'mean_birth_weights': Parameter(
            Types.LIST, 'list of mean birth weights from gestational age at birth 24-41 weeks'),
        'prob_disability_<28wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born at less than 28 weeks'),
        'prob_disability_28_32wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born between 28 and 32 weeks'),
        'prob_disability_33_36wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born between 33 and 36 weeks'),
        'prob_congenital_ba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_cba_type': Parameter(
            Types.LIST, 'Probability of types of CBA'),
        'odds_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline odds of a neonate developing sepsis following birth (early onset)'),
        'odds_ratio_sepsis_parity0': Parameter(
            Types.REAL, 'odds ratio for developing neonatal sepsis if this is the mothers first child'),
        'odds_ratio_sepsis_preterm': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for preterm neonates'),
        'odds_ratio_sepsis_lbw': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for low birth weight neonates'),
        'odds_ratio_sepsis_vlbw': Parameter(
            Types.REAL, 'odds ratio of developing sepsis for very low birth weight neonates'),
        'prob_sepsis_disabilities': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates who experience early onset sepsis '),
        'prompt_treatment_effect_sepsis': Parameter(
            Types.REAL, 'effect of prompt sepsis treatment on reducing mortality'),
        'delayed_treatment_effect_sepsis': Parameter(
            Types.REAL, 'effect of delayed sepsis treatment on reducing mortality'),
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
        'prob_mild_enceph_disabilities': Parameter(
            Types.LIST, 'list of prevalence levels of disability for neonates with mild encephalopathy'),
        'prob_mod_enceph_disabilities': Parameter(
            Types.LIST, 'list of prevalence levels of disability for neonates with moderate encephalopathy'),
        'prob_severe_enceph_disabilities': Parameter(
            Types.LIST, 'list of prevalence levels of disability for neonates with severe encephalopathy'),
        'prob_retinopathy_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing retinopathy of prematurity '),
        'prob_retinopathy_severity': Parameter(
            Types.LIST,
            'probabilities of severity of retinopathy'),
        'prob_early_breastfeeding_hb': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at home'),

        # ===================================== TREATMENT PARAMETERS ==================================================
        'prob_early_breastfeeding_hf': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at a health facility'),
        'prob_facility_offers_kmc': Parameter(
            Types.REAL, 'probability that the facility in which a low birth weight neonate is born will offer kangaroo'
                        ' mother care for low birth weight infants delivered at home'),
        'prob_successful_resuscitation': Parameter(
            Types.REAL, 'probability newborn resuscitation will be successful'),
        'rr_sepsis_tetracycline': Parameter(
            Types.REAL, 'relative risk of neonatal sepsis following tetracycline ointment treatment'),
        'rr_sepsis_cord_care': Parameter(
            Types.REAL, 'relative risk of neonatal sepsis following chlorhexidine cord care'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_encephalopathy': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to encephalopathy'),
        'cfr_failed_to_transition': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'cfr_preterm_birth': Parameter(
            Types.REAL, 'case fatality rate for a neonate born prematurely'),
        'rr_preterm_death_steroids': Parameter(
            Types.REAL, 'relative risk of death for preterm neonates following administration of antenatal '
                        'corticosteroids'),
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
        'nb_preterm_birth_disab': Property(Types.CATEGORICAL, 'Disability associated with preterm delivery',
                                           categories=['none', 'mild_motor_and_cog', 'mild_motor', 'moderate_motor',
                                                       'severe_motor']),
        'nb_congenital_anomaly': Property(Types.CATEGORICAL, 'Congenital Anomalies: None, Orthopedic, Gastrointestinal,'
                                                             'Neurological, Cosmetic, Other',
                                          #  TODO: This is being review to maybe be its own small module
                                          categories=['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whether his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_treatment_for_neonatal_sepsis': Property(Types.CATEGORICAL, 'If this neonate has received treatment for '
                                                                        'neonatal sepsis, and how promptly',
                                                     categories=['none', 'prompt_treatment', 'delayed_treatment']),
        'nb_neonatal_sepsis_disab': Property(Types.CATEGORICAL, 'Disability associated neonatal sepsis',
                                             categories=['none', 'mild_motor_and_cog', 'mild_motor',
                                                         'moderate_motor', 'severe_motor']),
        'nb_failed_to_transition': Property(Types.BOOL, 'whether this neonate has failed to transition to breathing on '
                                                        'their own following birth'),
        'nb_received_neonatal_resus': Property(Types.CATEGORICAL, 'If this neonate has received treatment for '
                                                                  'neonatal sepsis, and how promptly',
                                               categories=['none', 'prompt_treatment', 'delayed_treatment']),
        'nb_encephalopathy': Property(Types.CATEGORICAL, 'None, mild encephalopathy, moderate encephalopathy, '
                                                         'severe encephalopathy',
                                      categories=['none', 'mild_enceph', 'moderate_enceph', 'severe_enceph']),
        'nb_encephalopathy_disab': Property(Types.CATEGORICAL, 'Disability associated neonatal sepsis',
                                            categories=['none', 'mild_motor_and_cog', 'mild_motor',
                                                        'moderate_motor', 'severe_motor']),
        'nb_retinopathy_prem': Property(Types.CATEGORICAL, 'Level of visual disturbance due to retinopathy of'
                                                           ' prematurity: None, mild, moderate, severe, blindness',
                                        categories=['none', 'mild', 'moderate', 'severe', 'blindness']),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'extremely low birth weight (<1000g), '
                                                                  ' very low birth weight (<1500g), '
                                                                  'low birth weight (<2500g),'
                                                                  'normal birth weight (>2500g)',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),
        'nb_low_birth_weight': Property(Types.BOOL, 'whether this neonates weight is less than 2500g'),
        # TODO: this property is for the dx_test to recognise infants with low birth weight
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'size for gestational age categories',
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
        params = self.parameters

        if 'HealthBurden' in self.sim.modules.keys():
            params['nb_daly_weights'] = {
                'mild_motor_cognitive_preterm': self.sim.modules['HealthBurden'].get_daly_weight(357),
                'mild_motor_preterm': self.sim.modules['HealthBurden'].get_daly_weight(371),
                'moderate_motor_preterm': self.sim.modules['HealthBurden'].get_daly_weight(378),
                'severe_motor_preterm': self.sim.modules['HealthBurden'].get_daly_weight(383),
                # n.b. DALY weight for prematurity are separated by disability and gestation (<28wks, 28-32wks etc) but
                # the weight doesnt differ by gestation, only severity- so has been condensed here

                'mild_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(404),
                'moderate_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(405),
                'severe_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(402),
                'blindness_rptb': self.sim.modules['HealthBurden'].get_daly_weight(386),

                'mild_motor_cognitive_enceph': self.sim.modules['HealthBurden'].get_daly_weight(419),
                'mild_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(416),
                'moderate_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(411),
                'severe_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(410),

                'severe_infection_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(436),
                'mild_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(431),
                'moderate_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(438),
                'severe_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(435),
                'mild_motor_cognitive_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(441)}

        #  TODO: DALYs associated with congenital anomalies not yet applied, see comment above.

# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # All linear equations used in this module are stored within the nb_newborn_equations parameter below
        # TODO: predictors commented out presently for analysis

        params = self.parameters
        params['nb_newborn_equations'] = {

            'neonatal_sepsis': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['odds_early_onset_neonatal_sepsis'],
             Predictor('cord_care_given', external=True).when(True, params['rr_sepsis_cord_care']),
             Predictor('tetra_cycline_given', external=True).when(True, params['rr_sepsis_tetracycline'])),
            # Predictor('la_parity').when('0', params['odds_ratio_sepsis_parity0']),
            # Predictor('nb_early_preterm').when(True, params['odds_ratio_sepsis_preterm']),
            # Predictor('nb_late_preterm').when(True, params['odds_ratio_sepsis_preterm']),
            # Predictor('nb_low_birth_weight_status').when('low_birth_weight', params['odds_ratio_sepsis_lbw']).when(
            #    'very_low_birth_weight', params['odds_ratio_sepsis_vlbw']).when('extremely_low_birth_weight', params[
            #       'odds_ratio_sepsis_vlbw'])),


            'neonatal_sepsis_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_neonatal_sepsis'],
             Predictor('nb_treatment_for_neonatal_sepsis')
             .when('prompt_treatment', params['prompt_treatment_effect_sepsis'])
             .when('delayed_treatment', params['delayed_treatment_effect_sepsis'])),

            'encephalopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['odds_encephalopathy']),
            # Predictor('nb_early_onset_neonatal_sepsis').when(True,  params['odds_enceph_neonatal_sepsis']),
            # Predictor('sex').when('M',  params['odds_enceph_males']),
            # Predictor('mother_obstructed_labour', external=True).when(True, params['odds_enceph_obstructed_labour']),
            # Predictor().when('(la_obstructed_labour == True) & (la_obstructed_labour_treatment == "none")',
            #                 params['odds_enceph_obstructed_labour']),
            # Predictor('mother_gestational_htn', external=True).when(True, params['odds_enceph_hypertension']),
            # Predictor('mother_mild_pre_eclamp', external=True).when(True, params['odds_enceph_hypertension']),
            # Predictor('mother_severe_pre_eclamp', external=True).when(True, params['odds_enceph_hypertension'])),

            'encephalopathy_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_encephalopathy'],
             # Predictor('nb_encephalopathy').when('moderate', 2).when('severe', 3),
             Predictor('nb_received_neonatal_resus').when(True, 0.7)),

            'failure_to_transition': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_failure_to_transition']),

            'failed_to_transition_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_failed_to_transition'],
             Predictor('nb_received_neonatal_resus').when(True, 0.7)),

            'retinopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_retinopathy_preterm']),

            'preterm_birth_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_preterm_birth'],
             Predictor('received_antenatal_steroids', external=True).when('True', params['rr_preterm_death_steroids'])),

            'care_seeking_for_complication': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_care_seeking_for_complication'])
            }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_preterm_birth_disab'] = 'none'
        df.loc[df.is_alive, 'nb_congenital_anomaly'] = 'none'
        df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_treatment_for_neonatal_sepsis'] = 'none'
        df.loc[df.is_alive, 'nb_neonatal_sepsis_disab'] = 'none'
        df.loc[df.is_alive, 'nb_failed_to_transition'] = False
        df.loc[df.is_alive, 'nb_received_neonatal_resus'] = 'none'
        df.loc[df.is_alive, 'nb_encephalopathy'] = 'none'
        df.loc[df.is_alive, 'nb_encephalopathy_disab'] = 'none'
        df.loc[df.is_alive, 'nb_retinopathy_prem'] = 'none'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_low_birth_weight'] = False
        df.loc[df.is_alive, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.loc[df.is_alive, 'nb_early_breastfeeding'] = False
        df.loc[df.is_alive, 'nb_kangaroo_mother_care'] = False
        df.loc[df.is_alive, 'nb_death_after_birth'] = False
        df.loc[df.is_alive, 'nb_death_after_birth_date'] = pd.NaT

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):
        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=364))

        # Set the values of the complication tracker used by the logging event
        self.NewbornComplicationTracker = {'neonatal_sepsis': 0,
                                           'mild_enceph': 0,
                                           'moderate_enceph': 0,
                                           'severe_enceph': 0,
                                           'failed_to_transition': 0,
                                           't_e_d': 0,
                                           'resus': 0,
                                           'sep_treatment': 0,
                                           'vit_k': 0,
                                           'death': 0,
                                           'failed_to_transition_death': 0,
                                           'neonatal_sepsis_death': 0,
                                           'preterm_birth_death': 0,
                                           }

        # =======================Register dx_tests for complications during labour/postpartum=======================
        # We define the diagnostic tests that will be called within the health system interactions. As with Labour, the
        # sensitivity of these tests varies between facility type

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
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_low_birth_weight_hc=DxTest(
                property='nb_low_birth_weight',
                sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hc'], ))
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_low_birth_weight_hp=DxTest(
                property='nb_low_birth_weight',
                sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hp'], ))

    # ===================================== HELPER AND TESTING FUNCTIONS ==============================================
    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination. For equations that require external variables, they are also defined here"""
        params = self.parameters
        df = self.sim.population.props
        nci = self.newborn_care_info
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        mother_id = df.loc[person_id, 'mother_id']
        person = df.loc[[person_id]]

        # When calculating probability of neonatal sepsis an external intercept is used, this is the neonates ongoing
        # risk of sepsis (manipulated by interventions in labour and delivery), it is converted to odds for the equation
        if eq == params['nb_newborn_equations']['neonatal_sepsis']:

            cord_care_given = nci[person_id]['cord_care']
            tetra_cycline_given = nci[person_id]['tetra_eye_d']

            return self.rng.random_sample(size=1) < eq.predict(person, cord_care_given=cord_care_given,
                                                               tetra_cycline_given=tetra_cycline_given)[person_id]

        # Similarly when determine if a child will become encephalopathic, external variables stored in the dataframe
        # for the mother, are used as predictors
        if eq == params['nb_newborn_equations']['encephalopathy']:
            # obstructed_labour = df.at[mother_id, 'la_obstructed_labour']
            gestational_htn = df.at[mother_id, 'ps_gestational_htn']
            mild_pre_eclampsia = df.at[mother_id, 'ps_mild_pre_eclamp']
            severe_pre_eclampsia = df.at[mother_id, 'ps_severe_pre_eclamp']

            return self.rng.random_sample(size=1) < eq.predict(person, # mother_obstructed_labour=obstructed_labour,
                                                               mother_gestational_htn=gestational_htn,
                                                               mother_mild_pre_eclamp=mild_pre_eclampsia,
                                                               mother_severe_pre_eclamp=severe_pre_eclampsia)[person_id]

        if eq == params['nb_newborn_equations']['preterm_birth_death']:
            steroid_status = mni[mother_id]['corticosteroids_given']
            return self.rng.random_sample(size=1) < eq.predict(person, received_antenatal_steroids=steroid_status)[
                person_id]

        else:
            # If there are no external variables in the equation the calculation is as follows
            return self.rng.random_sample(size=1) < eq.predict(df.loc[[person_id]])[person_id]

    def set_birth_weight(self, child_id, gestation_at_birth):
        """This function generates a birth weight for each newborn, drawn randomly from a normal distribution around a
        mean birth weight for each gestational age in weeks. Babies below the 10th percentile for their gestational age
        are small for gestational age"""

        params = self.parameters
        df = self.sim.population.props

        # TODO: We dont have data for mean birth weight after 41 weeks, so currently convert gestational age of >41 to
        #  41
        if gestation_at_birth > 41:
            gestation_at_birth = 41

        # Mean birth weights for each gestational age are listed in a parameter starting at 24 weeks
        # We select the correct mean birth weight from the parameter
        mean_birth_weight_list_location = gestation_at_birth - 24

        # We randomly draw this newborns weight from a normal distribution around the mean for their gestation
        birth_weight = np.random.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=20)
        birth_weight_distribution = np.random.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location],
                                                     scale=20, size=10000) 
        # TODO: Unsure what value to use for the size parameter of the ND random draw, number of newborns of that
        #  gestation born on average per year?

        # Then we calculate the 10th and 90th percentile, these are the case definition for 'small for gestational age'
        # and 'large for gestational age'
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

        if birth_weight < 2500:  # TODO: we will eventually want output for VLBW and ELBW infants also
            df.at[child_id, 'nb_low_birth_weight'] = True

        if birth_weight < small_for_gestational_age_cutoff:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'small_for_gestational_age'
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
            self.NewbornComplicationTracker[f'{cause}_death'] += 1

            logger.debug(F'This is NewbornOutcomes scheduling a death for person %d on date %s who died due to {cause}'
                         'complications following birth', individual_id, self.sim.date)

    def on_birth(self, mother_id, child_id):
        """The on_birth function of this module sets properties of the newborn, including prematurity status, schedules
        functions to set weight and size and determines if this newborn will develop any complications following
        Additionally it schedules care for neonates who delivered in a facility or care seeking
        for home births and schedules the death event"""

        df = self.sim.population.props
        params = self.parameters
        nci = self.newborn_care_info

        # We check that the baby has survived labour and has been delivered (even if the mother did no survive)
        if (df.at[mother_id, 'is_alive'] and ~df.at[mother_id, 'la_intrapartum_still_birth']) or \
           (~df.at[mother_id, 'is_alive'] and df.at[mother_id, 'la_maternal_death_in_labour'] and
           ~df.at[mother_id, 'la_intrapartum_still_birth']):
            mni = self.sim.modules['Labour'].mother_and_newborn_info
            m = mni[mother_id]

        df.at[child_id, 'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_preterm_birth_disab'] = 'none'
        df.at[child_id, 'nb_congenital_anomaly'] = 'none'
        df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = False
        df.at[child_id, 'nb_treatment_for_neonatal_sepsis'] = 'none'
        df.at[child_id, 'nb_neonatal_sepsis_disab'] = 'none'
        df.at[child_id, 'nb_failed_to_transition'] = False
        df.at[child_id, 'nb_received_neonatal_resus'] = 'none'
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_encephalopathy_disab'] = 'none'
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child_id, 'nb_low_birth_weight'] = False
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

        if m['labour_state'] == 'late_preterm_labour':
            df.at[child_id, 'nb_late_preterm'] = True

        # Check no children born at term or postterm women are incorrectly categorised as preterm
        if m['labour_state'] == 'term_labour':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']
        if m['labour_state'] == 'postterm_labour':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']

        # Here we apply the prevalence of congenital birth anomalies in infants who have survived to delivery
        # TODO: this will be replaced by further work on CA
        if child.is_alive and ~m['stillbirth_in_labour']:
            if self.rng.random_sample() < params['prob_congenital_ba']:
                etiology = ['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']
                probabilities = params['prob_cba_type']
                random_choice = self.rng.choice(etiology, size=1, p=probabilities)
                df.at[child_id, 'nb_congenital_anomaly'] = random_choice

        # For all newborns we first generate a dictionary that will store the prophylactic interventions then receive at
        # birth if delivered in a facility and additional information

            # Set up NCI dictionary
            nci[child_id] = {'ga_at_birth': df.at[mother_id, 'ps_gestational_age_in_weeks'],
                             'cord_care': False,
                             'vit_k': False,
                             'tetra_eye_d': False,
                             'proph_abx': False,
                             'delivery_attended':  mni[mother_id]['delivery_attended'],
                             'delivery_setting': mni[mother_id]['delivery_setting'],
                             'sought_care_for_complication': False}

    # =================================== BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE ===================================
            # The set birth weight function is used to determine this newborns birth weight and size for gestational age
            self.set_birth_weight(child_id, nci[child_id]['ga_at_birth'])

    # ================================== COMPLICATIONS FOLLOWING BIRTH ================================================
            # Here, using linear equations, we determine individual newborn risk of complications following delivery.

            # ------------------------------------------- SEPSIS.... ------------------------------------------------
            # If the child's mother has delivered at home, we immediately apply the risk and make changes to the
            # data frame. Otherwise this is done during the HSI
            if m['delivery_setting'] == 'home_birth':
                if self.eval(params['nb_newborn_equations']['neonatal_sepsis'], child_id):
                    df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True
                    self.NewbornComplicationTracker['neonatal_sepsis'] += 1

                    logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            child_id, self.sim.date)

            # ---------------------------------------- ENCEPHALOPATHY.... --------------------------------------------
            # Term neonates then have a risk of encephalopathy applied
            if ~child.nb_early_preterm & ~child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['encephalopathy'], child_id):

                    # For a newborn who is encephalopathic we then set the severity
                    severity_enceph = self.rng.choice(('mild', 'moderate', 'severe'), p=params['prob_enceph_severity'])
                    if severity_enceph == 'mild':
                        df.at[child_id, 'nb_encephalopathy'] = 'mild_enceph'
                        self.NewbornComplicationTracker['mild_enceph'] += 1

                    elif severity_enceph == 'moderate':
                        df.at[child_id, 'nb_encephalopathy'] = 'moderate_enceph'
                        self.NewbornComplicationTracker['moderate_enceph'] += 1

                    else:
                        df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'
                        self.NewbornComplicationTracker['severe_enceph'] += 1

                    # Check all encephalopathy cases receive a grade
                    assert df.at[child_id, 'nb_encephalopathy'] != 'none'

            # -------------------------------------- RETINOPATHY... ------------------------------------------
            # ... and preterm neonates have a risk of retinopathy applied
            if (child.nb_early_preterm or child.nb_late_preterm) and self.eval(
             params['nb_newborn_equations']['retinopathy'], child_id):

                random_draw = self.rng.choice(('mild', 'moderate', 'severe', 'blindness'),
                                              p=params['prob_retinopathy_severity'])
                df.at[child_id, 'nb_retinopathy_prem'] = random_draw
                logger.debug(f'Neonate %d has developed {random_draw }retinopathy of prematurity',
                             child_id)

            # --------------------------------------- FAILURE TO TRANSITION... ---------------------------------------
            # Following the application of the above complications, we determine which neonates will fail to transition
            # (i.e. not spontaneously begin breathing) on birth

            if df.at[child_id, 'nb_encephalopathy'] != 'none':
                # All encephalopathic children will need some form of neonatal resuscitation
                df.at[child_id, 'nb_failed_to_transition'] = True
                self.NewbornComplicationTracker['failed_to_transition'] += 1

                logger.debug(f'Neonate %d has failed to transition from foetal to neonatal state and is not breathing '
                             f'following delivery', child_id)

            # Otherwise we use the linear model to calculate risk and make changes to the data frame
            elif self.eval(params['nb_newborn_equations']['failure_to_transition'], child_id):
                df.at[child_id, 'nb_failed_to_transition'] = True
                self.NewbornComplicationTracker['failed_to_transition'] += 1
                logger.debug(f'Neonate %d has failed to transition from foetal to neonatal state and is not breathing '
                             f'following delivery', child_id)

    # ======================================= SCHEDULING NEWBORN CARE  ================================================
        # Neonates who were delivered in a facility are automatically scheduled to receive care after birth

            if m['delivery_setting'] == 'health_centre' or m['delivery_setting'] == 'hospital':
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

            # TODO: Should preterm birth be considered a complication that may trigger care seeking?
            if (m['delivery_setting'] == 'home_birth') & (child.nb_failed_to_transition or
                                                          child.nb_early_onset_neonatal_sepsis or
                                                          child.nb_encephalopathy != 'none'):
                if self.eval(params['nb_newborn_equations']['care_seeking_for_complication'], child_id):
                    nci[child_id]['sought_care_for_complication'] = True

                    event = HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(
                        self, person_id=child_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        event, priority=0, topen=self.sim.date, tclose=self.sim.date + DateOffset(days=1))

                    logger.debug('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesSkilledAttendance'
                                 'FollowingBirthFacilityLevel1 for child %d whose mother has sought care after a '
                                 'complication has developed following a home_birth', child_id)

    # ============================================ BREAST FEEDING AT HOME ============================================
            # Using DHS data we apply a one of probability that women who deliver at home will initiate breastfeeding
            # within one hour of birth
            # We assume that neonates with complications will not start breastfeeding within one hour
            if (m['delivery_setting'] == 'home_birth') and (~child.nb_failed_to_transition and
                                                            ~child.nb_early_onset_neonatal_sepsis and
                                                            (child.nb_encephalopathy == 'none')):

                if self.rng.random_sample() < params['prob_early_breastfeeding_hb']:
                    df.at[child_id, 'nb_early_breastfeeding'] = True
                    logger.debug(
                        'Neonate %d has started breastfeeding within 1 hour of birth', child_id)
                else:
                    logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', child_id)

    # ========================================== SCHEDULING NEWBORN DEATH EVENT  ======================================
            # All newborns are then scheduled to pass through a newborn death event to determine likelihood of death
            # in the presence of complications

            self.sim.schedule_event(NewbornDeathEvent(self, child_id), self.sim.date + DateOffset(days=3))
            logger.info('This is NewbornOutcomesEvent scheduling NewbornDeathEvent for person %d', child_id)

    def on_hsi_alert(self, person_id, treatment_id):

        logger.info('This is NewbornOutcomes, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        logger.debug('This is Newborn Outcomes reporting my health values')

        df = self.sim.population.props
        p = self.parameters['nb_daly_weights']

        # Disability properties are mapped to DALY weights and stored for the health burden module

        health_values_1 = df.loc[df.is_alive, 'nb_retinopathy_prem'].map(
                    {'none': 0, 'mild': p['mild_vision_rptb'], 'moderate': p['moderate_vision_rptb'],
                     'severe': p['severe_vision_rptb'], 'blindness': p['blindness_rptb']})
        health_values_1.name = 'Retinopathy of Prematurity'

        health_values_2 = df.loc[df.is_alive, 'nb_encephalopathy_disab'].map(
            {'none': 0, 'mild_motor': p['mild_motor_enceph'], 'mild_motor_and_cog': p['mild_motor_cognitive_enceph'],
             'moderate_motor': p['moderate_motor_enceph'], 'severe_motor': p['severe_motor_enceph']})
        health_values_2.name = 'Neonatal Encephalopathy'

        health_values_3 = df.loc[df.is_alive, 'nb_neonatal_sepsis_disab'].map(
            {'none': 0, 'mild_motor': p['mild_motor_sepsis'], 'mild_motor_and_cog': p['mild_motor_cognitive_sepsis'],
             'moderate_motor': p['moderate_motor_sepsis'], 'severe_motor': p['severe_motor_sepsis']})
        health_values_3.name = 'Neonatal Sepsis Long term Disability'

        health_values_4 = df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'].map(
            {False: 0, True: p['severe_infection_sepsis']})
        health_values_4.name = 'Neonatal Sepsis Acute Disability'

        health_values_5 = df.loc[df.is_alive, 'nb_preterm_birth_disab'].map(
            {'none': 0, 'mild_motor': p['mild_motor_preterm'], 'mild_motor_and_cog': p['mild_motor_cognitive_preterm'],
             'moderate_motor': p['moderate_motor_preterm'], 'severe_motor': p['severe_motor_preterm']})
        health_values_5.name = 'Preterm Birth Disability'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive]], axis=1)

        scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
                          health_values_df.sum(axis=1)).fillna(1.0)
        health_values_df = health_values_df.multiply(scaling_factor, axis=0)

        return health_values_df


class NewbornDeathEvent(Event, IndividualScopeEventMixin):
    """This is the NewbornDeathEvent. It is scheduled for all newborns via the on_birth function. This event determines
    if those newborns who have experienced any complications will die due to them. This event will likely be changed as
     at present we dont deal with death of newborns due to congenital anomalies awaiting discussion regarding ongoing
     newborn care (NICU)"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        child = df.loc[individual_id]
        mni = self.sim.modules['Labour'].mother_and_newborn_info

        # Check the correct amount of time has passed between birth and the death event
        assert (self.sim.date - df.at[individual_id, 'date_of_birth']) == pd.to_timedelta(3, unit='D')

        # Using the set_neonatal_death_status function, defined above, it is determined if newborns who have experienced
        # complications will die because of them.

        # TODO: Main issue- as ALL encephalopathic neonates also experience failure to transition, they shouldnt go
        #  through both death equations? need to combine? or just have the risk of death applied once?

        if child.nb_early_onset_neonatal_sepsis:
            self.module.set_neonatal_death_status(individual_id, cause='neonatal_sepsis')

    #    if child.nb_encephalopathy != 'none':
    #        self.module.set_neonatal_death_status(individual_id, cause='encephalopathy')

        if child.nb_failed_to_transition:
            self.module.set_neonatal_death_status(individual_id, cause='failed_to_transition')

        if child.nb_early_preterm or child.nb_late_preterm:
            self.module.set_neonatal_death_status(individual_id, cause='preterm_birth')

        child = df.loc[individual_id]
        if child.nb_death_after_birth:
            self.module.NewbornComplicationTracker['death'] += 1
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)
        else:
            # Surviving newborns are scheduled to the disability event which determines morbidity following birth
            # complications
            self.sim.schedule_event(NewbornDisabilityEvent(self.module, individual_id), self.sim.date + DateOffset(
                days=4))
            logger.info('This is NewbornOutcomesEvent scheduling NewbornDisabilityEvent for person %d', individual_id)

        #  TODO: Tim C suggested we need to create an offset (using a distribution?) so we're generating deaths for the
        #   first 48 hours
        # TODO: use append to add cause of death to mni?

        # We now delete the MNI dictionary for mothers who have died in labour but their children have survived, this
        # is done here as we use variables from the mni as predictors in some of the above equations
        mother_id = df.loc[individual_id, 'mother_id']
        if df.at[mother_id, 'la_maternal_death_in_labour']:
            del mni[mother_id]


class NewbornDisabilityEvent(Event, IndividualScopeEventMixin):
    """ This is NewbornDisabilityEvent. It is scheduled by NewbornDeathEvent for surviving neonates. This event
    determines level of disability for neonates who have experience complications around the time of birth. The
    resulting disabilities are then mapped to DALY weights and reported to the health burden module."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        child = df.loc[individual_id]
        params = self.module.parameters
        nci = self.module.newborn_care_info

        if child.is_alive:
            # Neonates who dont develop any complications do not accrue any DALYs
            if ~child.nb_early_preterm and ~child.nb_late_preterm and child.nb_encephalopathy \
            == 'none' and ~child.nb_early_onset_neonatal_sepsis and ~child.nb_failed_to_transition:
                logger.debug('This is NewbornDisabilityEvent, person %d has not accrued any DALYs following delivery',
                             individual_id)
            else:
                logger.debug('This is NewbornDisabilityEvent, person %d will have their DALYs calculated following '
                             'complications after birth', individual_id)

            disability_categories = ['none', 'mild_motor_and_cog', 'mild_motor', 'moderate_motor', 'severe_motor']

            # Probability of preterm neonates developing long-lasting disability is organised by gestational age
            if nci[individual_id]['ga_at_birth'] < 28:
                df.at[individual_id, 'nb_preterm_birth_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                        p=params['prob_disability_'
                                                                                                 '<28wks'])
            if 27 < nci[individual_id]['ga_at_birth'] < 33:
                df.at[individual_id, 'nb_preterm_birth_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                        p=params['prob_disability_'
                                                                                                 '28_32wks'])
            if 32 < nci[individual_id]['ga_at_birth'] < 37:
                df.at[individual_id, 'nb_preterm_birth_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                        p=params['prob_disability_'
                                                                                                 '33_36wks'])
            if child.nb_early_onset_neonatal_sepsis:
                df.at[individual_id, 'nb_neonatal_sepsis_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                          p=params['prob_sepsis_'
                                                                                                   'disabilities'])
            # TODO: this may be too complex for available data?
            if child.nb_encephalopathy == 'mild_enceph':
                df.at[individual_id, 'nb_encephalopathy_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                         p=params['prob_mild_enceph_'
                                                                                                  'disabilities'])
            if child.nb_encephalopathy == 'moderate_enceph':
                df.at[individual_id, 'nb_encephalopathy_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                         p=params['prob_mod_enceph_'
                                                                                                  'disabilities'])
            if child.nb_encephalopathy == 'severe_enceph':
                df.at[individual_id, 'nb_encephalopathy_disab'] = self.module.rng.choice(disability_categories, size=1,
                                                                                         p=params['prob_severe_enceph_'
                                                                                                  'disabilities'])

        # TODO: ISSUE- DALY weight conditions are the same for each complication, a neonate could develop impairment
        #  from multiple complications. Currently we would map these weights cummulitively for a neonate, is that
        #  correct?

        # TODO: Congenital anomaly DALYs = > 300 different individual weights - to discuss with the team (a high number
        #  to do with congenital hear anomalies which we dont model) - maybe to be its own little module.


class NewbornDiseaseResetEvent(Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        # TODO: Is an event like this needed for this module, depends on how we're logging cases. Not all complications
        #  in this module resolve

# ================================ HEALTH SYSTEM INTERACTION EVENTS ================================================
        # TODO: HSIs that purely take up inpatient time? i.e. blank HSI that fires consecutively to show IP time


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
        the_appt_footprint['InpatientDays'] = 1  # TODO: confirm best appt footprint to use

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
        # This function manages the administration of essential newborn care and prophylaxis

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

            # ------------------------------------------ CORD CARE -------------------------------------------------
            # Consumables for cord care are recorded in the labour module, as part of the skilled birth attendance
            # package therefore aren't defined here on conditioned on
            # TODO: think I'm applying the effect of cord care twice, reduction in neonatal sepsis risk may be covered
            #  by 'clean birth practices' effect in labour module. review
            nci[person_id]['cord_care'] = True
            # Tetracycline eye care and vitamin k prophylaxis are conditioned on the availability of consumables

            # ------------------------------------------ TETRACYCLINE-----------------------------------------------
            if outcome_of_request_for_consumables['Item_Code'][item_code_tetracycline]:
                logger.debug('Neonate %d has received tetracycline eye drops to reduce sepsis risk following a facility'
                             'delivery', person_id)
                nci[person_id]['tetra_eye_d'] = True
                self.module.NewbornComplicationTracker['t_e_d'] += 1

            else:
                logger.debug('This facility has no tetracycline and therefore was not given')

            # ------------------------------------------ VITAMIN K --------------------------------------------------
            if outcome_of_request_for_consumables['Item_Code'][item_code_vit_k] and \
               outcome_of_request_for_consumables['Item_Code'][item_code_vit_k_syringe]:
                logger.debug('Neonate %d has received vitamin k prophylaxis following a facility delivery', person_id)
                nci[person_id]['vit_k'] = True
                self.module.NewbornComplicationTracker['vit_k'] += 1

                # TODO: this variable currently doesnt affect any outcomes - maybe effect could live in 'newborn first
                #  month module'
            else:
                logger.debug('This facility has no vitamin K and therefore was not given')

            # ---------------------------------EARLY INITIATION OF BREAST FEEDING -----------------------------------
            # A probably that early breastfeeding will initiated in a facility is applied
            if self.module.rng.random_sample() < params['prob_early_breastfeeding_hf']:
                df.at[person_id, 'nb_early_breastfeeding'] = True
                logger.debug('Neonate %d has started breastfeeding within 1 hour of birth', person_id)
            else:
                logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', person_id)
                # TODO: should initiation of breastfeeding vary between health centres and hospitals

            # TODO: Determine if neonates are given antibiotics for maternal risk factors

        # If this neonates delivery is attended, the function will run
        if nci[person_id]['delivery_attended']:
            prophylactic_interventions()
        else:
            # Otherwise they receives no benefit of prophylaxis
            logger.debug('neonate %d received no prophylaxis as they were delivered unattended', person_id)

        # --------------------------------- RECALCULATE SEPSIS RISK -------------------------------------

        # Following the administration of prophylaxis we determine if this neonate will develop sepsis
        if self.module.eval(params['nb_newborn_equations']['neonatal_sepsis'], person_id):
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True
            self.module.NewbornComplicationTracker['neonatal_sepsis'] += 1

            logger.debug('Neonate %d has developed early onset sepsis following a facility delivery on date %s',
                         person_id, self.sim.date)
            logger.info('%s|early_onset_nb_sep|%s', self.sim.date, {'person_id': person_id})

        # This function manages kangaroo mother care for low birth weight neonates
        def kangaroo_mother_care(facility_type):
            # We determine if staff will correctly identify if the neonate is low birth weight, and then initiate KMC
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_low_birth_weight_'
            f'{facility_type}', hsi_event=self):
                df.at[person_id, 'nb_kangaroo_mother_care'] = True
                self.module.NewbornComplicationTracker['kmc'] += 1

                logger.debug('Neonate %d has been correctly identified as being low birth weight, and kangaroo mother '
                             'care has been initiated', person_id, self.sim.date)
                # TODO: KMC reduces risk of sepsis at discharge. Maybe shouldn't be applied here?

        # Only stable neonates are assessed for KMC as per guidelines
        if nci[person_id]['delivery_attended'] and (~df.at[person_id, 'nb_early_onset_neonatal_sepsis'] and
                                                    ~df.at[person_id, 'nb_failed_to_transition'] and
                                                    df.at[person_id, 'nb_encephalopathy'] != 'none'):
            # Likelihood of correct assessment and treatment varied by facility type
            if nci[person_id]['delivery_setting'] == 'health_centre':
                kangaroo_mother_care('hc')
            elif nci[person_id]['delivery_setting'] == 'health_centre':
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
                # TODO: outcome_of_request_for_consumables is always false here, so commented out for now
                # if outcome_of_request_for_consumables:
                if nci[person_id]['delivery_attended']:
                    df.at[person_id, 'nb_received_neonatal_resus'] = 'prompt_treatment'
                    self.module.NewbornComplicationTracker['resus'] += 1

                else:
                    df.at[person_id, 'nb_received_neonatal_resus'] = 'delayed_treatment'
                    self.module.NewbornComplicationTracker['resus'] += 1

        if nci[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_initiation_of_neonatal_resus('hc')
        if nci[person_id]['delivery_setting'] == 'hospital':
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

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_neonatal_sepsis_'
                   f'{facility_type}', hsi_event=self):
                    # TODO: as above
                    #if outcome_of_request_for_consumables:
                    if nci[person_id]['delivery_attended']:
                            df.at[person_id, 'nb_treatment_for_neonatal_sepsis'] = 'prompt_treatment'
                            self.module.NewbornComplicationTracker['sep_treatment'] += 1

                    else:
                        df.at[person_id, 'nb_treatment_for_neonatal_sepsis'] = 'delayed_treatment'
                        self.module.NewbornComplicationTracker['sep_treatment'] += 1

        if nci[person_id]['delivery_setting'] == 'health_centre':
            assessment_and_treatment_newborn_sepsis('hc')
        if nci[person_id]['delivery_setting'] == 'hospital':
            assessment_and_treatment_newborn_sepsis('hp')

        # ------------------------------ (To go here- referral for further care) ---------------------------------------

class HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel2(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel2. This event is scheduled by
    the on_birth function of this module, automatically for neonates who delivered in facility and via a care seeking
    equation for those delivered at home. This event manages initial care of newborns following birth at level 2
    facilities. This event also houses assessment and treatment of complications following delivery (sepsis and failure
    to transition). Eventually it will scheduled additional treatment for those who need it"""

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
            self.module.NewbornComplicationTracker['neonatal_sepsis'] += 1

            logger.info('Neonate %d has developed early onset sepsis following a facility delivery on date %s',
                        person_id, self.sim.date)
            logger.info('%s|early_onset_nb_sep|%s', self.sim.date, {'person_id': person_id})

        self.module.kangaroo_mother_care('hp')
        self.module.assessment_and_initiation_of_neonatal_resus('hp')
        self.module.assessment_and_treatment_newborn_sepsis('hp')

        # ------------------------------ (To go here- referral for further care) ---------------------------------------


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is NewbornOutcomesLoggingEvent. The event runs every year and calculates yearly incidence of key outcomes
    following birth, and logs them in a dataframe to be used by analysis files """
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        # Denominators...
        dfx = pd.to_datetime(df['date_of_birth'])
        yearly_births = len(dfx.index[dfx.dt.year == self.sim.date.year])

        newborn_deaths = len(df.index[df.nb_death_after_birth & (df.nb_death_after_birth_date > one_year_prior) &
                                      (df.nb_death_after_birth_date < self.sim.date)])

        early_preterm_births = len(df.index[df.nb_early_preterm & (df.date_of_birth > one_year_prior) &
                                            (df.date_of_birth < self.sim.date)])

        late_preterm_births = len(df.index[df.nb_late_preterm & (df.date_of_birth > one_year_prior) &
                                           (df.date_of_birth < self.sim.date)])

        total_preterm_births = early_preterm_births + late_preterm_births

        low_birth_weight = len(df.index[(df.nb_low_birth_weight_status != 'normal_birth_weight') & (df.date_of_birth >
                                                                                                    one_year_prior) &
                                        (df.date_of_birth < self.sim.date)])

        small_for_gestational_age = len(df.index[(df.nb_size_for_gestational_age == 'small_for_gestational_age') &
                                                 (df.date_of_birth > one_year_prior) & (df.date_of_birth <
                                                                                        self.sim.date)])

        # yearly number of complications
        sepsis = self.module.NewbornComplicationTracker['neonatal_sepsis']
        mild_enceph = self.module.NewbornComplicationTracker['mild_enceph']
        mod_enceph = self.module.NewbornComplicationTracker['moderate_enceph']
        severe_enceph = self.module.NewbornComplicationTracker['severe_enceph']
        all_enceph = mild_enceph + mod_enceph + severe_enceph
        ftt = self.module.NewbornComplicationTracker['failed_to_transition']
        death = self.module.NewbornComplicationTracker['death']
        sepsis_death = self.module.NewbornComplicationTracker['death']
        ftt_death = self.module.NewbornComplicationTracker['death']
        preterm_death = self.module.NewbornComplicationTracker['death']

        sepsis_treatment = self.module.NewbornComplicationTracker['sep_treatment']
        resus = self.module.NewbornComplicationTracker['resus']
        tetra_cycline = self.module.NewbornComplicationTracker['t_e_d']

        dict_for_output = {'neonatal_deaths': newborn_deaths,
                           'checker_deaths':death,
                           'sepsis_deaths_crude':sepsis_death,
                           'ftt_deaths_crude':ftt_death,
                           'preterm_deaths':preterm_death,
                           'nmr_early': newborn_deaths/yearly_births * 1000,
                           'early_preterm_births': early_preterm_births,
                           'late_preterm_births': late_preterm_births,
                           'total_preterm_births': total_preterm_births,
                           'tptb_incidence':total_preterm_births/yearly_births *100,
                           'low_birth_weight': low_birth_weight / yearly_births * 100,
                           'small_for_gestational_age': small_for_gestational_age / yearly_births * 100,
                           'sepsis_crude': sepsis,
                           'sepsis_incidence': sepsis / yearly_births * 100,
                           'sepsis_treatment_crude': sepsis_treatment,
                           'sepsis_treatment_rate': sepsis_treatment/sepsis * 100,
                           'mild_enceph_incidence': mild_enceph / yearly_births * 100,
                           'mod_enceph_incidence': mod_enceph / yearly_births * 100,
                           'severe_enceph_incidence': severe_enceph / yearly_births * 100,
                           'total_enceph_incidence': all_enceph / yearly_births * 100,
                           'ftt_crude': ftt,
                           'ftt_incidence': ftt / yearly_births * 100,
                           'resus_crude': resus,
                           'resus_rate': resus / ftt * 100,
                           }
        # TODO: health system outputs, check denominators

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        # Reset the EventTracker
        self.NewbornComplicationTracker = {'neonatal_sepsis': 0,
                                           'mild_enceph': 0,
                                           'moderate_enceph': 0,
                                           'severe_enceph': 0,
                                           'failed_to_transition': 0,
                                           't_e_d': 0,
                                           'resus': 0,
                                           'sep_treatment': 0,
                                           'vit_k': 0,
                                           'death': 0,
                                           'failed_to_transition_death': 0,
                                           'neonatal_sepsis_death': 0,
                                           'preterm_birth_death': 0,
                                           }

