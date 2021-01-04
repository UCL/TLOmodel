"""
The Newborn Outcomes Module

Overview:
Represents a neonates first 24-48 hours of life.

Key Definitions


# Things to note:
    * The model is not accurately paremeterised as per 06/12/2020
    * Calibration has not been completed- deaths have been calibrated to GBD estimates

# todo: finish
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.postnatal_supervisor import HSI_PostnatalSupervisor_NeonatalWardInpatientCare
# from tlo.methods.hiv import HSI_Hiv_TestAndRefer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewbornOutcomes(Module):
    """This module is responsible for the key conditions/complications experienced by a neonate immediately following
    birth. As the focus of this module is day 0-1 of a newborns life any complications experienced by neonates in the
     later neonatal period (defined here as day 2- day 28) are managed in the PostnatalSupervisorModule. Key conditions
     included in this module are early onset neonatal infection and sepsis, neonatal encephalopathy,
    complications of prematurity and failure to transition. This module also manages care of the newborn during facility
    deliveries and referral to higher level care for unwell newborns"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath

        # This dictionary will store information related to the neonates delivery that does not need to be stored in
        # the main data frame
        self.newborn_care_info = dict()

        # This dictionary is used to count each occurrence of an 'event' of interest. These stored counts are used
        # in the LoggingEvent to calculate key outcomes (i.e. incidence rates, neonatal mortality rate etc)
        self.newborn_complication_tracker = dict()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.USES_SYMPTOMMANAGER
    }

    PARAMETERS = {
        'mean_birth_weights': Parameter(
            Types.LIST, 'list of mean birth weights from gestational age at birth 24-41 weeks'),
        'standard_deviation_birth_weights': Parameter(
            Types.LIST, 'list of standard deviations associated with mean birth weights from gestational age at '
                        'birth 24-41 weeks'),
        'prob_disability_<28wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born at less than 28 weeks'),
        'prob_disability_28_32wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born between 28 and 32 weeks'),
        'prob_disability_33_36wks': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates born between 33 and 36 weeks'),
        'prob_congenital_ba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_early_onset_neonatal_sepsis_day_0': Parameter(
            Types.REAL, 'baseline probability of a neonate developing early onset sepsis between birth and 24hrs'),
        'prob_sepsis_disabilities': Parameter(
            Types.LIST, 'list of prevalence of levels of disability for neonates who experience early onset sepsis '),
        'prob_failure_to_transition': Parameter(
            Types.REAL, 'baseline probability of a neonate developing intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'prob_respiratory_distress_preterm': Parameter(
            Types.REAL, 'probability that a preterm infant will experience respiratory distress at birth'),
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
        'prob_breastfeeding_type': Parameter(
            Types.LIST, 'probabilities that a woman is 1.) not breastfeeding 2.) non-exclusively breastfeeding '
                        '3.)exclusively breastfeeding at birth (until 6 months)'),

        # ===================================== TREATMENT PARAMETERS ==================================================
        'prob_early_breastfeeding_hf': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at a health facility'),
        'prob_facility_offers_kmc': Parameter(
            Types.REAL, 'probability that the facility in which a low birth weight neonate is born will offer kangaroo'
                        ' mother care for low birth weight infants delivered at home'),
        'treatment_effect_inj_abx_sep': Parameter(
            Types.REAL, 'effect of injectable antibiotics treatment on reducing mortality from sepsis'),
        'treatment_effect_supp_care_sep': Parameter(
            Types.REAL, 'effect of full supportive care treatment on reducing mortality from sepsis'),
        'treatment_effect_cord_care': Parameter(
            Types.REAL, 'effect of full supportive care treatment on reducing incidence of sepsis'),
        'treatment_effect_clean_birth': Parameter(
            Types.REAL, 'effect of clean birth practices on reducing incidence of sepsis'),
        'treatment_effect_early_init_bf': Parameter(
            Types.REAL, 'effect of early initiation of breastfeeding on reducing incidence of sepsis'),
        'treatment_effect_resuscitation': Parameter(
            Types.REAL, 'effect of resuscitation on newborn mortality associated with encephalopathy'),
        'treatment_effect_resuscitation_preterm': Parameter(
            Types.REAL, 'effect of delayed resuscitation on newborn mortality associated with prematurity'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_mild_enceph': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to mild encephalopathy'),
        'cfr_moderate_enceph': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to moderate encephalopathy'),
        'cfr_severe_enceph': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to severe encephalopathy'),
        'cfr_failed_to_transition': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'cfr_preterm_birth': Parameter(
            Types.REAL, 'case fatality rate for a neonate born prematurely'),
        'rr_preterm_death_early_preterm': Parameter(
            Types.REAL, 'relative risk of preterm death in early preterm neonates '),
        'treatment_effect_steroid_preterm': Parameter(
            Types.REAL, 'relative risk of death for preterm neonates following administration of antenatal '
                        'corticosteroids'),
        'treatment_effect_kmc': Parameter(
            Types.REAL, 'treatment effect of kangaroo mother care on preterm mortality'),
        'cfr_congenital_anomaly': Parameter(
            Types.REAL, 'case fatality rate for congenital anomalies'),
        'cfr_rds_preterm': Parameter(
            Types.REAL, 'case fatality rate for respiratory distress syndrome of prematurity'),
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
        'nb_congenital_anomaly': Property(Types.BOOL, 'Whether this newborn has been born with a congenital anomaly'),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whether his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_inj_abx_neonatal_sepsis': Property(Types.BOOL, 'If this neonate has injectable antibiotics as treatment '
                                                           'for neonatal sepsis'),
        'nb_supp_care_neonatal_sepsis': Property(Types.BOOL, 'If this neonate has received full supportive care for '
                                                             'neonatal sepsis (in hospital)'),
        'nb_neonatal_sepsis_disab': Property(Types.CATEGORICAL, 'Disability associated neonatal sepsis',
                                             categories=['none', 'mild_motor_and_cog', 'mild_motor',
                                                         'moderate_motor', 'severe_motor']),
        'nb_preterm_respiratory_distress': Property(Types.BOOL, 'whether this preterm newborn has respiratory '
                                                                'distress syndrome (RDS)'),
        'nb_not_breathing_at_birth': Property(Types.BOOL, 'whether this neonate has failed to transition to breathing '
                                                          'on their own following birth'),
        'nb_received_neonatal_resus': Property(Types.BOOL, 'If this neonate has received resuscitation'),
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
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'size for gestational age categories',
                                                categories=['small_for_gestational_age', 'average_for_gestational_age',
                                                            'large_for_gestational_age']),
        'nb_early_init_breastfeeding': Property(Types.BOOL, 'whether this neonate initiated breastfeeding '
                                                            'within 1 hour of birth '),
        'nb_breastfeeding_type': Property(Types.CATEGORICAL, 'How this neonate is being breastfed',
                                          categories=['none', 'non_exclusive', 'exclusive']),
        'nb_kangaroo_mother_care': Property(Types.BOOL, 'whether this neonate received kangaroo mother care following '
                                                        'birth'),
        'nb_clean_birth': Property(Types.BOOL, 'whether this neonate received clean birth practices at deliver'),
        'nb_received_cord_care': Property(Types.BOOL, 'whether this neonate received chlorhexidine cord care'),
        'nb_death_after_birth': Property(Types.BOOL, 'whether this child has died following complications after birth'),
        'nb_death_after_birth_date': Property(Types.DATE, 'date on which the child died after birth'),
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_NewbornOutcomes.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        # Here we map 'disability' parameters to associated DALY weights to be passed to the health burden module
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

                'mild_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(431),
                'moderate_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(438),
                'severe_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(435),
                'mild_motor_cognitive_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(441)}

        # ======================================= LINEAR MODEL EQUATIONS ===========================
        # All linear equations used in this module are stored within the nb_newborn_equations
        # parameter below

        # TODO: process of 'selection' of important predictors in linear equations is ongoing, a linear model that
        #  is empty of predictors at the end of this process will be converted to a set probability

        params = self.parameters
        params['nb_newborn_equations'] = {

            # This equation is used to determine a newborns risk of early onset neonatal sepsis
            'early_onset_neonatal_sepsis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_early_onset_neonatal_sepsis_day_0'],
                Predictor('nb_clean_birth').when('True', params['treatment_effect_clean_birth']),
                Predictor('nb_received_cord_care').when('True', params['treatment_effect_cord_care']),
                Predictor('nb_early_init_breastfeeding').when(True, params['treatment_effect_early_init_bf'])),


            # This equation is used to determine a newborns risk of encephalopathy
            'encephalopathy': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_encephalopathy']),

            # This equation is used to determine a preterm newborns risk of respiratory distress syndrome
            # (incomplete lung development)
            'rds_preterm': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_respiratory_distress_preterm']),

            # This equation is used to determine a newborns risk of failing to transition after delivery,
            # triggering resuscitation
            'not_breathing_at_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_failure_to_transition']),

            # This equation is used to determine a premature newborns risk of retinopathy (a/w predictors)
            'retinopathy': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_retinopathy_preterm']),

            # This equation is used to determine the probability that a the mother of a newborn, who has been delivered
            # at home, will seek care in the event that a newborn experiences complications after birth
            'care_seeking_for_complication': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_care_seeking_for_complication']),

            # This equation is used to determine a preterm newborns risk of death due to 'complications of prematurity'
            #  not explicitly modelled here (therefore excluding sepsis, encephalopathy and RDS)
            'preterm_birth_other_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_preterm_birth'],
                Predictor('nb_early_preterm').when(True, params['rr_preterm_death_early_preterm']),
                Predictor('nb_kangaroo_mother_care').when(True, params['treatment_effect_kmc']),
                Predictor('received_corticosteroids', external=True).when('True', params['treatment_effect_steroid_'
                                                                                         'preterm'])),

            # This equation is used to determine a the risk of death for a newborn who doesnt breathe at birth.
            'not_breathing_at_birth_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_failed_to_transition'],
                Predictor('nb_received_neonatal_resus').when(True, params['treatment_effect_resuscitation'])),

            # Theses equations are used to determine the risk of death for encephalopathic newborns.
            'mild_enceph_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_mild_enceph'],
                Predictor('nb_received_neonatal_resus').when(True, params['treatment_effect_resuscitation'])),
            'moderate_enceph_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_moderate_enceph'],
                Predictor('nb_received_neonatal_resus').when(True, params['treatment_effect_resuscitation'])),
            'severe_enceph_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_severe_enceph'],
                Predictor('nb_received_neonatal_resus').when(True, params['treatment_effect_resuscitation'])),

            # This equation is used to determine a newborns risk of death from sepsis
            'neonatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_neonatal_sepsis'],
                Predictor('nb_inj_abx_neonatal_sepsis').when(True, params['treatment_effect_inj_abx_sep']),
                Predictor('nb_supp_care_neonatal_sepsis').when(True, params['treatment_effect_supp_care_sep'])),

            'congenital_anomaly_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_congenital_anomaly']),

            'respiratory_distress_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_rds_preterm'],
                Predictor('nb_received_neonatal_resus').when(True, params['treatment_effect_resuscitation_preterm']))}

    def initialise_population(self, population):
        df = population.props

        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_preterm_birth_disab'] = 'none'
        df.loc[df.is_alive, 'nb_congenital_anomaly'] = False
        df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_inj_abx_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_supp_care_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_neonatal_sepsis_disab'] = 'none'
        df.loc[df.is_alive, 'nb_preterm_respiratory_distress'] = False
        df.loc[df.is_alive, 'nb_not_breathing_at_birth'] = False
        df.loc[df.is_alive, 'nb_received_neonatal_resus'] = False
        df.loc[df.is_alive, 'nb_encephalopathy'] = 'none'
        df.loc[df.is_alive, 'nb_encephalopathy_disab'] = 'none'
        df.loc[df.is_alive, 'nb_retinopathy_prem'] = 'none'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.loc[df.is_alive, 'nb_early_init_breastfeeding'] = False
        df.loc[df.is_alive, 'nb_breastfeeding_type'] = 'none'
        df.loc[df.is_alive, 'nb_kangaroo_mother_care'] = False
        df.loc[df.is_alive, 'nb_clean_birth'] = False
        df.loc[df.is_alive, 'nb_received_cord_care'] = False
        df.loc[df.is_alive, 'nb_death_after_birth'] = False
        df.loc[df.is_alive, 'nb_death_after_birth_date'] = pd.NaT

    def initialise_simulation(self, sim):
        # Register logging event
        sim.schedule_event(NewbornOutcomesLoggingEvent(self), sim.date + DateOffset(years=1))

        # Set the values of the complication tracker used by the logging event (and described above)
        self.newborn_complication_tracker = {
            'early_init_bf': 0,
            'early_onset_sepsis': 0,
            'neonatal_sepsis_death': 0,
            'mild_enceph': 0,
            'mild_enceph_death': 0,
            'moderate_enceph': 0,
            'moderate_enceph_death': 0,
            'severe_enceph': 0,
            'severe_enceph_death': 0,
            'preterm_rds': 0,
            'respiratory_distress_death': 0,
            'congenital_anomaly_death': 0,
            'not_breathing_at_birth': 0,
            'not_breathing_at_birth_death': 0,
            'preterm_birth_other_death': 0,
            't_e_d': 0,
            'resus': 0,
            'sep_treatment': 0,
            'vit_k': 0,
            'death': 0}

        # =======================Register dx_tests for complications during labour/postpartum=======================
        # As with the labour module these dx_tests represent a probability that one of the following clinical outcomes
        # will be detected by the health care worker and treatment will be initiated

        # The sensitivity parameter varies by facility type, with hc meaning health centre and hp meaning hospital

        dx_manager = sim.modules['HealthSystem'].dx_manager

        # Neonatal sepsis...
        dx_manager.register_dx_test(assess_neonatal_sepsis_hc=DxTest(
            property='nb_early_onset_neonatal_sepsis',
            sensitivity=self.parameters['sensitivity_of_assessment_of_neonatal_sepsis_hc']))
        dx_manager.register_dx_test(assess_neonatal_sepsis_hp=DxTest(
            property='nb_early_onset_neonatal_sepsis',
            sensitivity=self.parameters['sensitivity_of_assessment_of_neonatal_sepsis_hp']))

        # Not breathing at birth ...
        dx_manager.register_dx_test(assess_not_breathing_at_birth_hc=DxTest(
            property='nb_not_breathing_at_birth',
            sensitivity=self.parameters['sensitivity_of_assessment_of_ftt_hc']))
        dx_manager.register_dx_test(assess_not_breathing_at_birth_hp=DxTest(
            property='nb_not_breathing_at_birth',
            sensitivity=self.parameters['sensitivity_of_assessment_of_ftt_hp']))

        # Low birth weight...
        dx_manager.register_dx_test(assess_low_birth_weight_hc=DxTest(
            property='nb_low_birth_weight_status', target_categories=['extremely_low_birth_weight',
                                                                      'very_low_birth_weight',
                                                                      'low_birth_weight'],
            sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hc'])),
        dx_manager.register_dx_test(assess_low_birth_weight_hp=DxTest(
            property='nb_low_birth_weight_status', target_categories=['extremely_low_birth_weight',
                                                                      'very_low_birth_weight',
                                                                      'low_birth_weight'],
            sensitivity=self.parameters['sensitivity_of_assessment_of_lbw_hp']))

    # ===================================== HELPER AND TESTING FUNCTIONS ===========================
    def eval(self, eq, person_id):
        """
        This function compares the result of a specific linear equation with a random draw providing
        a boolean for the outcome under examination. For equations that require external variables,
        they are also defined here
        :param eq: The linear model equation being evaluated
        :param person_id: person_id
        :return: BOOL outcome
        """
        df = self.sim.population.props
        nci = self.newborn_care_info
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        mother_id = df.loc[person_id, 'mother_id']
        person = df.loc[[person_id]]

        # Here we define all the possible external variables used in the linear model
        steroid_status = mni[mother_id]['corticosteroids_given']

        # We return a BOOLEAN
        return self.rng.random_sample(size=1) < eq.predict(person, received_corticosteroids=steroid_status,
                                                           )[person_id]

    # ========================================= OUTCOME FUNCTIONS  ===================================================
    # These functions are called within the on_birth function or
    # HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant depending on location of delivery. Generally the
    # output an individuals probability of an outcome (complication, disability, death) and make the relevant changes
    # to the data frame

    def set_birth_weight(self, child_id, gestation_at_birth):
        """
        This function generates a birth weight for each newborn, drawn randomly from a normal
        distribution around a mean birth weight for each gestational age in weeks. Babies below the
        10th percentile for their gestational age are classified as small for gestational age. Other cut offs for birth
        weight status are evident from the code below
        :param child_id: child_id
        :param gestation_at_birth: The gestational age, in weeks, of this newborn at birth
        """
        params = self.parameters
        df = self.sim.population.props

        # TODO: the functionality will be simplified to consider antenatal predictors and effect of interventions
        #  (balanced protein supplementation) - use linear model (dont know how best to capture pregnancy experience)

        # Mean birth weights for each gestational age are listed in a parameter starting at 24 weeks
        # We select the correct mean birth weight from the parameter
        mean_birth_weight_list_location = int(min(41, gestation_at_birth) - 24)
        standard_deviation = params['standard_deviation_birth_weights'][mean_birth_weight_list_location]

        # We randomly draw this newborns weight from a normal distribution around the mean for their gestation
        birth_weight = np.random.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location],
                                        scale=standard_deviation)

        # Then we calculate the 10th and 90th percentile, these are the case definition for 'small for gestational age'
        # and 'large for gestational age'
        small_for_gestational_age_cutoff = scipy.stats.norm.ppf(
            0.1, loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=standard_deviation)

        large_for_gestational_age_cutoff = scipy.stats.norm.ppf(
            0.9, loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=standard_deviation)

        # Make the appropriate changes to the data frame
        if birth_weight >= 2500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        elif 1500 <= birth_weight < 2500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'low_birth_weight'
            logger.debug(key='message', data=f'Child {child_id} has been born low birth weight')

        elif 1000 <= birth_weight < 1500:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'very_low_birth_weight'
            logger.debug(key='message', data=f'Child {child_id} has been born very low birth weight')

        elif birth_weight < 1000:
            df.at[child_id, 'nb_low_birth_weight_status'] = 'extremely_low_birth_weight'
            logger.debug(key='message', data=f'Child {child_id} has been born extremely low birth weight')

        if birth_weight < small_for_gestational_age_cutoff:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'small_for_gestational_age'
            logger.debug(key='message', data=f'Child {child_id} has been born small for gestational age')

        elif birth_weight > large_for_gestational_age_cutoff:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'large_for_gestational_age'
            logger.debug(key='message', data=f'Child {child_id} has been born large for gestational age')

        else:
            df.at[child_id, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
            logger.debug(key='message', data=f'Child {child_id} has been born average for gestational age')

    def apply_risk_of_neonatal_infection_and_sepsis(self, child_id):
        """
        This function uses the linear model to determines if a neonate will develop early onset neonatal sepsis.
        It is called during the on_birth function or during HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant
        dependent on delivery setting.
        :param child_id: child_id
        """
        params = self.parameters
        df = self.sim.population.props

        # The linear model calculates the individuals probability of early_onset_neonatal_sepsis
        if self.eval(params['nb_newborn_equations']['early_onset_neonatal_sepsis'], child_id):
            self.newborn_complication_tracker['early_onset_sepsis'] += 1

            df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True
            logger.debug(key='message', data=f'Neonate {child_id} has developed early onset sepsis following delivery')

    def apply_risk_of_encephalopathy(self, child_id):
        """
        This function uses the linear model to determines if a neonate will develop neonatal encephalopathy, at what
        severity, and makes the appropriate changes to the data frame. It is called during the on_birth function or
        during HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant dependent on delivery setting.
        :param child_id: child_id
        """
        params = self.parameters
        df = self.sim.population.props

        # The linear model calculates the individuals probability of encephalopathy
        if self.eval(params['nb_newborn_equations']['encephalopathy'], child_id):

            # For a newborn who is encephalopathic we then set the severity using a weighted probability derived from
            # the prevalence of severity of encephalopathy in the encephalopathic population
            severity_enceph = self.rng.choice(('mild', 'moderate', 'severe'), p=params['prob_enceph_severity'])
            if severity_enceph == 'mild':
                df.at[child_id, 'nb_encephalopathy'] = 'mild_enceph'
                self.newborn_complication_tracker['mild_enceph'] += 1
            elif severity_enceph == 'moderate':
                df.at[child_id, 'nb_encephalopathy'] = 'moderate_enceph'
                self.newborn_complication_tracker['moderate_enceph'] += 1
            else:
                df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'
                self.newborn_complication_tracker['severe_enceph'] += 1

            # Check all encephalopathy cases receive a grade
            assert df.at[child_id, 'nb_encephalopathy'] != 'none'

    def apply_risk_of_preterm_respiratory_distress_syndrome(self, child_id):
        """
        This function uses the linear model to determine if a preterm neonate will develop respiratory distress
        syndrome. It is called during the on_birth function or during
        HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant
        dependent on delivery setting.
        :param child_id: child_id
        """
        df = self.sim.population.props
        params = self.parameters
        child = df.loc[child_id]

        # Ensure only preterm infants have risk of RDS applied
        assert child.nb_early_preterm or child.nb_late_preterm

        # Use the linear model to calculate individual risk and make changes
        if self.eval(params['nb_newborn_equations']['rds_preterm'], child_id):
            df.at[child_id, 'nb_preterm_respiratory_distress'] = True
            self.newborn_complication_tracker['preterm_rds'] += 1
            logger.debug(key='message', data=f'Neonate {child_id} who was delivered preterm is experiencing '
                                             f'respiratory distress syndrome ')

    def apply_risk_of_not_breathing_at_birth(self, child_id):
        """
        This function uses the linear model to determines if a neonate will not sufficiently initiate breathing at birth
         and makes the appropriate changes to the data frame. It is called during the on_birth function or
        during HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant dependent on delivery setting.
        :param child_id: child_id
        """
        params = self.parameters
        df = self.sim.population.props

        # We assume all newborns with encephalopathy and respiratory distress syndrome will require some form of
        # resuscitation and will not be effectively breathing at birth
        if df.at[child_id, 'nb_encephalopathy'] != 'none' or df.at[child_id, 'nb_preterm_respiratory_distress']:
            df.at[child_id, 'nb_not_breathing_at_birth'] = True
            self.newborn_complication_tracker['not_breathing_at_birth'] += 1
            logger.debug(key='message', data=f'Neonate {child_id} is not breathing following delivery')

        # Otherwise we use the linear model to calculate risk of inadequete breathing due to other causes not
        # explicitly modelled
        elif self.eval(params['nb_newborn_equations']['not_breathing_at_birth'], child_id):
            df.at[child_id, 'nb_not_breathing_at_birth'] = True
            self.newborn_complication_tracker['not_breathing_at_birth'] += 1
            logger.debug(key='message', data=f'Neonate {child_id} is not breathing following delivery')

    def apply_risk_of_death_from_complication(self, individual_id, complication):
        """
        This function  is called for neonates that have experienced a complication after birth and
        determines if this complication will cause their death. Properties in the data frame are set
        accordingly, and the result of the equation is tracked. It is called during the on_birth function or during
        HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant dependent on delivery setting.
        :param individual_id: individual_id
        :param complication: complication that has put this newborn at risk of death. This complication has an
        associated linear model equation determining risk of death
        """
        df = self.sim.population.props
        params = self.parameters
        nci = self.newborn_care_info

        if self.eval(params['nb_newborn_equations'][f'{complication}_death'], individual_id):

            # If this newborn will die due to this complication we do not immediately schedule the
            # InstantaneousDeathEvent in this function as newborns with multiple complications may 'die' from more than
            # one complication (hence using this property)
            df.at[individual_id, 'nb_death_after_birth'] = True
            df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

            # The death is counted by the tracker
            self.newborn_complication_tracker[f'{complication}_death'] += 1

            # And we store this 'cause' of death in the nci dictionary
            nci[individual_id]['cause_of_death_after_birth'].append(complication)
            logger.debug(key='message', data=f'This is NewbornOutcomes scheduling a death for person {individual_id} '
                                             f'on who died due to {complication} complications following birth')

    def set_death_and_disability_status(self, individual_id):
        """
        This function cycles through each complication of which a newborn may die, if the newborn has experienced this
        complication it calls the apply_risk_of_death_from_complication function which uses the linear model to
        determine individual risk of death from the relevant complication. If the newborn 'dies' due to one or more
        complication then the InstantaneousDeathEvent is scheduled. For surviving newborns, probability of disability
        associated with complications is applied and PostnatalWeekOneEvent is scheduled. It is called during the
        on_birth function or during HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant dependent on delivery
        setting.
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        child = df.loc[individual_id]
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        nci = self.newborn_care_info
        params = self.parameters

        # Using the set_neonatal_death_status function, defined above, it is determined if newborns who have experienced
        # complications will die because of them.
        if child.nb_early_onset_neonatal_sepsis:
            self.apply_risk_of_death_from_complication(individual_id, complication='neonatal_sepsis')

        if child.nb_encephalopathy == 'mild_enceph':
            self.apply_risk_of_death_from_complication(individual_id, complication='mild_enceph')
        if child.nb_encephalopathy == 'moderate_enceph':
            self.apply_risk_of_death_from_complication(individual_id, complication='moderate_enceph')
        if child.nb_encephalopathy == 'severe_enceph':
            self.apply_risk_of_death_from_complication(individual_id, complication='severe_enceph')

        # As well as being at risk of death from sepsis or encephalopathy, preterm neonates are at risk of death from
        # respiratory distress syndrome and 'other' non-modelled causes
        if child.nb_early_preterm or child.nb_late_preterm:
            if child.nb_preterm_respiratory_distress:
                self.apply_risk_of_death_from_complication(individual_id, complication='respiratory_distress')
            self.apply_risk_of_death_from_complication(individual_id, complication='preterm_birth_other')

        # Death due to breathing difficulties is implicitly captured in the CFR for encephalopathy and pretmaturity/RDS,
        # therefore only neonates without those complications face a separate CFR for 'not breathing at birth'
        if child.nb_not_breathing_at_birth and child.nb_encephalopathy == 'none' and \
            (~child.nb_preterm_respiratory_distress):
            self.apply_risk_of_death_from_complication(individual_id, complication='not_breathing_at_birth')

        if child.nb_congenital_anomaly:
            self.apply_risk_of_death_from_complication(individual_id, complication='congenital_anomaly')

        # If a newborn has died, the death is scheduled and tracked
        if child.nb_death_after_birth:
            self.newborn_complication_tracker['death'] += 1
            self.sim.schedule_event(demography.InstantaneousDeath(self, individual_id,
                                                                  cause='neonatal'), self.sim.date)

        # If the newborn has survived and did not develop any complications during this period, they do not accrue any
        # DALYs
        else:
            if (~child.nb_early_preterm and ~child.nb_late_preterm and child.nb_encephalopathy == 'none' and
               ~child.nb_early_onset_neonatal_sepsis and ~child.nb_not_breathing_at_birth):
                logger.debug(key='message', data=f'Person {individual_id} has not accrued any DALYs following '
                                                 f'delivery')

            else:
                # Otherwise, we now will calculate the newborns disability
                logger.debug(key='message',
                             data=f'Child {individual_id} will have now their DALYs calculated following complications'
                                  f'after birth')

                # DALY weights for each newborn condition can be categorised as follows
                disability_categories = ['none',
                                         'mild_motor_and_cog',  # Mild motor plus cognitive impairments due to...
                                         'mild_motor',  # Mild motor impairment due to...
                                         'moderate_motor',  # Moderate motor impairment due to...
                                         'severe_motor']  # Severe motor impairment due to...

                # We use a probability weighted random choice to determine which disability 'category' each newborn
                # will develop due to their complication - probability of more severe disability will correspond with
                # more 'severe'conditions (i.e. very early preterm, severe encephalopathy)

                # Probability of preterm neonates developing long-lasting disability is organised by gestational age
                choice = self.rng.choice
                if nci[individual_id]['ga_at_birth'] < 28:
                    df.at[individual_id, 'nb_preterm_birth_disab'] = choice(disability_categories,
                                                                            p=params['prob_disability_<28wks'])
                if 27 <= nci[individual_id]['ga_at_birth'] < 33:
                    df.at[individual_id, 'nb_preterm_birth_disab'] = choice(disability_categories,
                                                                            p=params['prob_disability_28_32wks'])
                if 32 <= nci[individual_id]['ga_at_birth'] < 37:
                    df.at[individual_id, 'nb_preterm_birth_disab'] = choice(disability_categories,
                                                                            p=params['prob_disability_33_36wks'])
                if child.nb_early_onset_neonatal_sepsis:
                    df.at[individual_id, 'nb_neonatal_sepsis_disab'] = choice(disability_categories,
                                                                              p=params['prob_sepsis_disabilities'])
                if child.nb_encephalopathy == 'mild_enceph':
                    df.at[individual_id, 'nb_encephalopathy_disab'] = choice(disability_categories,
                                                                             p=params['prob_mild_enceph_disabilities'])
                if child.nb_encephalopathy == 'moderate_enceph':
                    df.at[individual_id, 'nb_encephalopathy_disab'] = choice(disability_categories,
                                                                             p=params['prob_mod_enceph_disabilities'])
                if child.nb_encephalopathy == 'severe_enceph':
                    df.at[individual_id, 'nb_encephalopathy_disab'] = choice(disability_categories,
                                                                             p=params[
                                                                                 'prob_severe_enceph_disabilities'])

                # TODO: currently a neonate could develop a life time DALY weight from more than one complication.
                #  They will carry both (or more) of these DALY weights for life. Is it right to assume that these
                #  weights are additive - in reality a newborn would not develop 'mild' impairment twice, two 'causes'
                #  of mild impairment would likely lead to more severe impairment (but maybe this method captures that?)

                # TODO: should complication variables be reset? A newborn will never be newborn again so...

                # TODO: disability for congenital anomaly

        # We now delete the MNI dictionary for mothers who have died in labour but their children have survived, this
        # is done here as we use variables from the mni as predictors in some of the above equations
        mother_id = df.loc[individual_id, 'mother_id']
        if df.at[mother_id, 'la_maternal_death_in_labour']:
            del mni[mother_id]

        del nci[individual_id]

    # ======================================== HSI INTERVENTION FUNCTIONS =============================================
    # These functions are called by HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant and broadly represent
    # interventions/packages of interventions to manage the main complications in the module

    def essential_newborn_care(self, hsi_event):
        """
        This function contains interventions delivered as part of 'essential newborn care'. These include clean birth
        practices, cord care, vitamin k and eye care
        :param hsi_event:  The HSI event in which this function is called
        (HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant)
        """
        df = self.sim.population.props
        nci = self.newborn_care_info
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        person_id = hsi_event.target
        mother_id = df.loc[person_id, 'mother_id']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # -------------------------------------- CLEAN BIRTH PRACTICES -----------------------------------------------
        # First we check if clean practices were observed during this babies birth, if so we store as a property within
        # the nci dictionary. It is used to reduce incidence of sepsis (consumables counted in labour)
        if mni[mother_id]['clean_birth_practices']:
            df.at[person_id, 'nb_clean_birth'] = True

        # -------------------------------------- CHLORHEXIDINE CORD CARE ----------------------------------------------
        # Next we determine if cord care with chlorhexidine is applied (consumables are counted during labour)
        df.at[person_id, 'nb_cord_care_received'] = True

        # ---------------------------------- VITAMIN D AND EYE CARE -----------------------------------------------
        # TODO: Currently, these interventions do not effect incidence or outcomes. To be reviewed.

        # We define the consumables
        item_code_tetracycline = pd.unique(
            consumables.loc[consumables['Items'] == 'Tetracycline eye ointment 1%_3.5_CMST', 'Item_Code'])[0]
        item_code_vit_k = pd.unique(
            consumables.loc[consumables['Items'] == 'vitamin K1  (phytomenadione) 1 mg/ml, 1 ml, inj._100_IDA',
                                                    'Item_Code'])[0]
        item_code_vit_k_syringe = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe,  disposable 2ml,  hypoluer with 23g needle_each_'
                                                    'CMST', 'Item_Code'])[0]
        consumables_vit_k_and_eye_care = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_tetracycline: 1, item_code_vit_k: 1, item_code_vit_k_syringe: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_vit_k_and_eye_care, to_log=True)

        # If they are available the intervention is delivered
        if outcome_of_request_for_consumables['Item_Code'][item_code_tetracycline]:

            logger.debug(key='message', data=f'Neonate {person_id} has received tetracycline eye drops following a '
                                             f'facility delivery')
            nci[person_id]['tetra_eye_d'] = True
            self.newborn_complication_tracker['t_e_d'] += 1
        else:
            logger.debug(key='message', data='This facility has no tetracycline and therefore was not given')

        # This process is repeated for vitamin K
        if outcome_of_request_for_consumables['Item_Code'][item_code_vit_k] and outcome_of_request_for_consumables[
                                                                                'Item_Code'][item_code_vit_k_syringe]:
            logger.debug(key='message', data=f'Neonate {person_id} has received vitamin k prophylaxis following'
                                             f' a facility delivery')
            nci[person_id]['vit_k'] = True
            self.newborn_complication_tracker['vit_k'] += 1
        else:
            logger.debug(key='message', data='This facility has no vitamin K and therefore was not given')

    def breast_feeding(self, person_id, birth_setting):
        """
        This function is used to set breastfeeding status for newborns. It schedules the BreastfeedingStatusUpdateEvent
        for breastfed newborns. It is called during the
        on_birth function or during HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant dependent on delivery
        setting.
        :param person_id: person_id
        :param birth_setting: hf (health facility) or hb (home birth)
        """
        df = self.sim.population.props
        params = self.parameters

        # First we determine the 'type' of breastfeeding this newborn will receive
        random_draw = self.rng.choice(('none', 'non_exclusive', 'exclusive'), p=params['prob_breastfeeding_type'])
        df.at[person_id, 'nb_breastfeeding_type'] = random_draw

        # Log that information
        if random_draw == 'none':
            logger.debug(key='message', data=f'Neonate {person_id} will not be breastfed')
        else:
            logger.debug(key='message', data=f'Neonate {person_id} will be {random_draw}ly breastfed up to 6 months')

        # For newborns that are breastfed, we apply a probability that breastfeeding was initiated within one hour. This
        # varies between home births and facility deliveries
        if df.at[person_id, 'nb_breastfeeding_type'] != 'none':
            if self.rng.random_sample() < params[f'prob_early_breastfeeding_{birth_setting}']:
                df.at[person_id, 'nb_early_init_breastfeeding'] = True
                self.newborn_complication_tracker['early_init_bf'] += 1

                logger.debug(key='message', data=f'Neonate {person_id} has started breastfeeding within 1 hour of '
                                                 f'birth')
            else:
                logger.debug(key='message', data=f'Neonate {person_id} did not start breastfeeding within 1 hour of '
                                                 f'birth')

            # For breastfed neonates we schedule a future event where breastfeeding status is updated
            self.sim.schedule_event(BreastfeedingStatusUpdateEventSixMonths(self, person_id),
                                    self.sim.date + DateOffset(months=6))

    def kangaroo_mother_care(self, hsi_event, facility_type):
        """
        This function manages the diagnosis and treatment of low birth weight neonates who have
        delivered in a facility. It is called by the HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant.
        The intervention delivered is Kangaroo Mother Care (KMC) which includes skin-to-skin nursing and encouragement
        of frequent and exclusive breastfeeding
        :param hsi_event: The HSI event in which this function is called
        (HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant)
        :param facility_type: health centre (hc) or hospital (hp)
        """
        df = self.sim.population.props
        person_id = hsi_event.target

        # We use the dx_manager to determine if a newborn who is low birth weight is correctly identified and treated
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_low_birth_weight_{facility_type}', hsi_event=hsi_event):
            # Store treatment as a property of the newborn used to apply treatment effect
            df.at[person_id, 'nb_kangaroo_mother_care'] = True

            logger.debug(key='message', data=f'Neonate {person_id} has been correctly identified as being low birth '
                                             f'weight, and kangaroo mother care has been initiated')

    def immunisations(self, child_id):
        """
        This is a placeholder. Will house/call immunisation events for newborns
        :param child_id: child_id
        """
        # TODO: Discuss with Tara
        pass

    def hiv_prophylaxis_and_referral(self, child_id):
        """
        This is a placeholder. Will house/call HIV testing for exposed newborns
        :param child_id: child_id
        """
        pass

        # TODO: link up future screening with PNC visit

    #    if 'Hiv' in self.sim.modules:
    #        self.sim.modules['HealthSystem'].schedule_hsi_event(
    #            HSI_Hiv_TestAndRefer(person_id=child_id, module=self.sim.modules['Hiv']),
    #            topen=self.sim.date,
    #            tclose=None,
    #            priority=0
    #        )

    #        self.sim.modules['HealthSystem'].schedule_hsi_event(
    #            HSI_Hiv_TestAndRefer(person_id=child_id, module=self.sim.modules['Hiv']),
    #                topen=self.sim.date + DateOffset(weeks=4),
    #                tclose=None,
    #                priority=0
    #                )

    def assessment_and_initiation_of_neonatal_resus(self, hsi_event, facility_type):
        """
        This function manages the diagnosis of failure to transition/encephalopathy and the
        administration of neonatal resuscitation for neonates delivered in a facility.
        It is called by the HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant.
        :param hsi_event: The HSI event in which this function is called
        (HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant)
        :param facility_type: health centre (hc) or hospital (hp)
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Required consumables are defined
        pkg_code_resus = pd.unique(consumables.loc[
                                       consumables['Intervention_Pkg'] == 'Neonatal resuscitation (institutional)',
                                       'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_resus: 1}, 'Item_Code': {}}

        # Query if the required consumables are available
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
           hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=False)

        # Use the dx_manager to determine if staff will correctly identify this neonate will require resuscitation
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
           dx_tests_to_run=f'assess_not_breathing_at_birth_{facility_type}', hsi_event=hsi_event):

            # Then, if the consumables are available,resuscitation is started. We assume this is delayed in
            # deliveries that are not attended
            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_resus]:
                df.at[person_id, 'nb_received_neonatal_resus'] = True

                # Log the consumables request
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=True)

                self.newborn_complication_tracker['resus'] += 1

    def assessment_and_treatment_newborn_sepsis(self, hsi_event, facility_type):
        """
        This function manages the treatment of early onset neonatal sepsis for neonates delivered in a facility.
         It is called by the
        HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant. Treatment for sepsis includes either injectable
         antibiotics or full supportive care (antibiotics, fluids, oxygen etc) and varies between facility level.
        :param hsi_event: The HSI event in which this function is called
        (HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant)
        :param facility_type: health centre (hc) or hospital (hp)
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # We assume that only hospitals are able to deliver full supportive care for neonatal sepsis, full supportive
        # care evokes a stronger treatment effect than injectable antibiotics alone

        if facility_type == 'hp':
            # Define the consumables
            pkg_code_sep = pd.unique(consumables.loc[
                                     consumables['Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                     'Intervention_Pkg_Code'])[0]

            consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

            # Query if the required consumables are available
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                 hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=False)

            # Use the dx_manager to determine if staff will correctly identify this neonate will treatment for sepsis
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                 dx_tests_to_run=f'assess_neonatal_sepsis_{facility_type}', hsi_event=hsi_event):

                # Then, if the consumables are available, treatment for sepsis is delivered
                if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
                    df.at[person_id, 'nb_supp_care_neonatal_sepsis'] = True
                    self.newborn_complication_tracker['sep_treatment'] += 1

        # The same pattern is then followed for health centre care
        elif facility_type == 'hc':
            item_code_iv_penicillin = pd.unique(
                consumables.loc[consumables['Items'] == 'Benzylpenicillin 1g (1MU), PFR_Each_CMST', 'Item_Code'])[0]
            item_code_iv_gentamicin = pd.unique(
                consumables.loc[consumables['Items'] == 'Gentamicin 40mg/ml, 2ml_each_CMST', 'Item_Code'])[0]
            item_code_giving_set = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with '
                                                                                     'needle',
                                                                                     'Item_Code'])[0]

            consumables_inj_abx_sepsis = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_iv_penicillin: 1, item_code_iv_gentamicin: 1,
                              item_code_giving_set: 1}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_inj_abx_sepsis, to_log=False)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
               dx_tests_to_run=f'assess_neonatal_sepsis_{facility_type}', hsi_event=hsi_event):

                if outcome_of_request_for_consumables:
                    df.at[person_id, 'nb_inj_abx_neonatal_sepsis'] = True
                    self.newborn_complication_tracker['sep_treatment'] += 1

    def hsi_cannot_run(self, individual_id):
        """
        This function determines what happens to neonates if HSI_NewbornOutcomes_
        ReceivesSkilledAttendanceFollowingBirth cannot run with the current service configuration
        :param individual_id: individual_id
        """
        nci = self.newborn_care_info
        df = self.sim.population.props

        # We apply risk of complication onset and death within HSI_NewbornOutcomes_ ReceivesSkilledAttendanceFollowing
        # Birth for those who are born in facility (this allows risk to be modified to prophylaxis administered in
        # facility)

        # Therefore if this event cannot run we need to apply risk of complication developing, as done below
        if ~nci[individual_id]['sought_care_for_complication']:

            self.module.apply_risk_of_neonatal_infection_and_sepsis(individual_id)
            self.module.apply_risk_of_encephalopathy(individual_id)
            if df.at[individual_id, 'nb_early_preterm'] or df.at[individual_id, 'nb_late_preterm']:
                self.module.apply_risk_of_preterm_respiratory_distress_syndrome(individual_id)
            self.module.apply_risk_of_not_breathing_at_birth(individual_id)
            self.module.set_death_and_disability_status(individual_id)

    def on_birth(self, mother_id, child_id):
        """The on_birth function of this module sets key properties of all newborns, including prematurity
        status and schedules functions to set weight and size. For newborns delivered at home it determines if they will
        experience complications following birth (early onset sepsis, encephalopathy, failure to transition) and if
        these complications will lead to death or disability .For newborns delivered in facility it schedules
        HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant which represents the care newborns should receive
        after being delivered in a facility.
        :param mother_id: mother_id
        :param child_id: child_id
        """
        df = self.sim.population.props
        params = self.parameters
        nci = self.newborn_care_info

        # We check that the baby has survived labour and has been delivered (even if the mother did not survive)
        if (df.at[mother_id, 'is_alive'] and ~df.at[mother_id, 'la_intrapartum_still_birth']) or \
           (~df.at[mother_id, 'is_alive'] and df.at[mother_id, 'la_maternal_death_in_labour'] and
           ~df.at[mother_id, 'la_intrapartum_still_birth']):
            mni = self.sim.modules['Labour'].mother_and_newborn_info
            m = mni[mother_id]

        df.at[child_id, 'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_preterm_birth_disab'] = 'none'
        df.at[child_id, 'nb_congenital_anomaly'] = False
        df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = False
        df.at[child_id, 'nb_neonatal_sepsis_disab'] = 'none'
        df.at[child_id, 'nb_preterm_respiratory_distress'] = False
        df.at[child_id, 'nb_not_breathing_at_birth'] = False
        df.at[child_id, 'nb_received_neonatal_resus'] = False
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_encephalopathy_disab'] = 'none'
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child_id, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.at[child_id, 'nb_early_init_breastfeeding'] = False
        df.at[child_id, 'nb_breastfeeding_type'] = 'none'
        df.at[child_id, 'nb_kangaroo_mother_care'] = False
        df.at[child_id, 'nb_clean_birth'] = False
        df.at[child_id, 'nb_received_cord_care'] = False
        df.at[child_id, 'nb_death_after_birth'] = False
        df.at[child_id, 'nb_death_after_birth_date'] = pd.NaT

        child = df.loc[child_id]

        # 'Category' of prematurity (early/late) is stored as a tempory property of the mother via the MNI dictionary
        # generated in labour (this is because some interventions delivered to the mother are based on prematurity)

        # We now store a newborns 'category of prematurity' within the main data frame
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

        if child.is_alive and ~m['stillbirth_in_labour']:

            #  Next we populate the newborn info dictionary with relevant parameters
            nci[child_id] = {'ga_at_birth': df.at[mother_id, 'ps_gestational_age_in_weeks'],
                             'vit_k': False,
                             'tetra_eye_d': False,
                             'proph_abx': False,
                             'delivery_attended': mni[mother_id]['delivery_attended'],
                             'delivery_setting': mni[mother_id]['delivery_setting'],
                             'sought_care_for_complication': False,
                             'cause_of_death_after_birth': []}

            # Check these variables are not unassigned
            assert nci[child_id]['delivery_attended'] != 'none'
            assert nci[child_id]['delivery_setting'] != 'none'

            # ====================================== COMPLICATIONS ====================================================
            # Determine if this child will be born with a congenital anomaly
            if self.rng.random_sample() < params['prob_congenital_ba']:
                df.at[child_id, 'nb_congenital_anomaly'] = True
                # TODO: replace with more detailed modelling

            # TODO: remove as function as only called here
            # The set birth weight function is used to determine this newborns birth weight and size for gestational age
            self.set_birth_weight(child_id, nci[child_id]['ga_at_birth'])

            # For all preterm newborns we apply a risk of retinopathy of prematurity
            # TODO: oxygen therapy is a significant predictor - apply this risk to facility deliver newborns after
            #  interventions to capture effect?
            if (child.nb_early_preterm or child.nb_late_preterm) and self.eval(
              params['nb_newborn_equations']['retinopathy'], child_id):

                # For newborns with retinopathy we then use a weighted random draw to determine the severity of the
                # retinopathy
                random_draw = self.rng.choice(('mild', 'moderate', 'severe', 'blindness'),
                                              p=params['prob_retinopathy_severity'])
                df.at[child_id, 'nb_retinopathy_prem'] = random_draw
                logger.debug(key='message', data=f'Neonate {child_id} has developed {random_draw} retinopathy of '
                                                 f'prematurity')

            # If the child's mother has delivered at home, we immediately apply the risk of complications and make
            # changes to the data frame. Otherwise this is done during the HSI
            if nci[child_id]['delivery_setting'] == 'home_birth':
                self.apply_risk_of_neonatal_infection_and_sepsis(child_id)
                self.apply_risk_of_encephalopathy(child_id)

                # Term neonates then have a risk of encephalopathy applied
                if child.nb_early_preterm or child.nb_late_preterm:
                    self.apply_risk_of_preterm_respiratory_distress_syndrome(child_id)

                # Following the application of the above complications, we determine if this newborn will or will not
                # breathe at birth
                self.apply_risk_of_not_breathing_at_birth(child_id)

                # And we see if this newborn will start breastfeeding
                self.breast_feeding(child_id, birth_setting='hb')  # hb = home birth

            # Neonates who were delivered in a facility are automatically scheduled to receive care after birth at the
            # same level of facility that they were delivered in
            if m['delivery_setting'] == 'health_centre':
                event = HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant(
                    self, person_id=child_id, facility_level_of_this_hsi=1)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message', data=f'This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_'
                                                 f'ReceivesSkilledAttendanceFollowingBirth for child '
                                                 f'{child_id} following a facility delivery')

            elif m['delivery_setting'] == 'hospital':
                event = HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant(
                    self, person_id=child_id, facility_level_of_this_hsi=int(self.rng.choice([1, 2])))
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message', data=f'This is NewbornOutcomesEvent scheduling '
                                                 f'HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant'
                                                 f'for child {child_id} following a facility delivery')

                # todo: we're randomly allocating type of facility here, which is wrong

            # ========================================== CARE SEEKING  ================================================
            # For newborns who have been delivered at home, but have immediately developed a complication, we apply a
            # probability that their mother will seek care and bring them to
            # HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant

            if (m['delivery_setting'] == 'home_birth') and (child.nb_not_breathing_at_birth or
                                                            child.nb_early_onset_neonatal_sepsis or
                                                            child.nb_encephalopathy != 'none'):
                if self.eval(params['nb_newborn_equations']['care_seeking_for_complication'], child_id):
                    nci[child_id]['sought_care_for_complication'] = True

                    event = HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant(
                        self, person_id=child_id, facility_level_of_this_hsi=int(self.rng.choice([1, 2])))

                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        event, priority=0, topen=self.sim.date, tclose=self.sim.date + DateOffset(days=1))

                    logger.debug(key='message', data=f'This is NewbornOutcomesEvent scheduling '
                                                     f'HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant'
                                                     f'FacilityLevel1 for child {child_id }whose mother has sought care'
                                                     f'after a complication has developed following a home_birth')

                else:
                    # If care will not be sought for this newborn we immediately apply risk of death and make changes to
                    # the data frame accordingly
                    self.set_death_and_disability_status(child_id)

    def on_hsi_alert(self, person_id, treatment_id):
        logger.info(key='message', data=f'This is NewbornOutcomes, being alerted about a health system interaction '
                                        f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        """
        This function reports the DALY weights for this module generated in the previous month
        :return: data frame containing the DALY weights
        """
        logger.debug(key='message', data='This is Newborn Outcomes reporting my health values')

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

        health_values_4 = df.loc[df.is_alive, 'nb_preterm_birth_disab'].map(
            {'none': 0, 'mild_motor': p['mild_motor_preterm'], 'mild_motor_and_cog': p['mild_motor_cognitive_preterm'],
             'moderate_motor': p['moderate_motor_preterm'], 'severe_motor': p['severe_motor_preterm']})
        health_values_4.name = 'Preterm Birth Disability'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive]], axis=1)

        scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
                          health_values_df.sum(axis=1)).fillna(1.0)
        health_values_df = health_values_df.multiply(scaling_factor, axis=0)

        return health_values_df
        # TODO: health burden module currently isnt registered as the scaling factor above doesnt keep values > 1


class HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant. This event is scheduled by
    the on_birth function of this module, automatically for neonates who delivered in facility and via a care seeking
    equation for those delivered at home. This event represents the care newborns should receive immediately following
    their delivery in a facility with a skilled birth attendant. This event also
    houses assessment and treatment of complications following delivery (sepsis and failure
    to transition). Eventually it will scheduled additional treatment for those who need it."""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesSkilledAttendance'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        # TODO: confirm best appointment footprint to use
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        nci = self.module.newborn_care_info
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        logger.info(key='message', data=f'This is HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendant:'
                                        f' child {person_id} is receiving care following delivery in a health facility')

        # ========================  PREVENTATIVE INTERVENTIONS DELIVERED AT/NEAR BIRTH ===============================
        if ~nci[person_id]['sought_care_for_complication']:
            self.module.essential_newborn_care(self)

            if nci[person_id]['delivery_setting'] == 'health_centre':
                self.module.kangaroo_mother_care(self, 'hc')
            elif nci[person_id]['delivery_setting'] == 'hospital':
                self.module.kangaroo_mother_care(self, 'hp')

            self.module.breast_feeding(person_id, birth_setting='hf')  # hf = 'health facility'
        else:
            # Otherwise they receives no benefit of prophylaxis
            logger.debug(key='message', data=f'neonate {person_id} did not receive essential newborn care')

        # =================================  COMPLICATION RISKS ===================================================
        # First we determine which, if any, complications this newborn will experience immediately following, or very
        # soon after birth

        if ~nci[person_id]['sought_care_for_complication']:
            self.module.apply_risk_of_neonatal_infection_and_sepsis(person_id)
            self.module.apply_risk_of_encephalopathy(person_id)
            if df.at[person_id, 'nb_early_preterm'] or df.at[person_id, 'nb_late_preterm']:
                self.module.apply_risk_of_preterm_respiratory_distress_syndrome(person_id)

            self.module.apply_risk_of_not_breathing_at_birth(person_id)

        # This function manages initiation of neonatal resuscitation
        if nci[person_id]['delivery_setting'] == 'health_centre':
                self.module.assessment_and_initiation_of_neonatal_resus(self, 'hc')
        if nci[person_id]['delivery_setting'] == 'hospital':
                self.module.assessment_and_initiation_of_neonatal_resus(self, 'hp')

        # This function manages the assessment and treatment of neonatal sepsis, and follows the same structure as
        # resuscitation
        if nci[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_newborn_sepsis(self, 'hc')
        if nci[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_newborn_sepsis(self, 'hp')

        # In the case of neonates who needed resuscitation, but do not receive the intervention, we apply a risk of
        # encephalopathy due to additional hypoxia at birth
        if (df.at[person_id, 'nb_encephalopathy'] == 'none') and \
            df.at[person_id, 'nb_not_breathing_at_birth'] and \
            ~df.at[person_id, 'nb_received_neonatal_resus']:
            self.module.apply_risk_of_encephalopathy(person_id)

        # =======================================  RISK OF DEATH ======================================================
        # For newborns that have experience any complications following delivery in a facility we now determine if
        # they will die following treatment
        if df.at[person_id, 'nb_early_onset_neonatal_sepsis'] or \
            df.at[person_id, 'nb_not_breathing_at_birth'] or \
            df.at[person_id, 'nb_encephalopathy'] != 'none':
            self.module.set_death_and_disability_status(person_id)

        if not df.at[person_id, 'nb_death_after_birth']:
            self.module.immunisations(person_id)
            self.module.hiv_prophylaxis_and_referral(person_id)

            # Surviving neonates with complications on day 1 are admitted to the inpatient event which lives in the
            # Postnatal Supervisor module
            # TODO : guidelines not very clear
            if df.at[person_id, 'nb_early_onset_neonatal_sepsis'] or df.at[person_id, 'nb_encephalopathy'] != 'none' \
                or df.at[person_id, 'nb_early_preterm'] or df.at[person_id, 'nb_late_preterm']:
                event = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                    self.sim.modules['PostnatalSupervisor'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='message', data=f'Neonate {person_id} did not receive care after birth as the squeeze factor '
                                         f'was too high')
        self.module.hsi_cannot_run(person_id)

        return False

    def not_available(self):
        person_id = self.target
        logger.debug(key='message', data=f'Neonate {person_id} did not receive care after birth as this HSI is not '
                                         f'allowed in current configuration')

        self.module.hsi_cannot_run(person_id)
        # ------------------------------ (To go here- referral for further care) --------------------


class BreastfeedingStatusUpdateEventSixMonths(Event, IndividualScopeEventMixin):
    """ This is BreastfeedingStatusUpdateEventSixMonths. It is scheduled via the breastfeeding function.
    Children who are alive and still breastfeeding by six months have their breastfeeding status updated. Those who will
    continue to breastfeed are scheduled BreastfeedingStatusUpdateEventTwoYears
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        child = df.loc[individual_id]

        if not child.is_alive:
            return

        if child.nb_breastfeeding_type == 'exclusive':
            random_draw = self.module.rng.choice(('non_exclusive', 'none'), p=[0.5, 0.5])
            df.at[individual_id, 'nb_breastfeeding_type'] = random_draw

        if child.nb_breastfeeding_type == 'non_exclusive':
            random_draw = self.module.rng.choice(('non_exclusive', 'none'), p=[0.5, 0.5])
            df.at[individual_id, 'nb_breastfeeding_type'] = random_draw

        if child.nb_breastfeeding_type != 'none':
            self.sim.schedule_event(BreastfeedingStatusUpdateEventTwoYears(self.module, individual_id),
                                    self.sim.date + DateOffset(months=18))


class BreastfeedingStatusUpdateEventTwoYears(Event, IndividualScopeEventMixin):
    """ This is BreastfeedingStatusUpdateEventTwoYears. It is scheduled via the breastfeeding function.
    Children who are alive and still breastfeeding by six months have their breastfeeding status updated. Those who will
    continue to breastfeed are scheduled BreastfeedingStatusUpdateEventTwoYears """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        if not df.at[individual_id, 'is_alive']:
            return

        df.at[individual_id, 'nb_breastfeeding_type'] = 'none'


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is NewbornOutcomesLoggingEvent. The event runs every year and calculates yearly
    incidence of key outcomes following birth, and logs them in a dataframe to be used by analysis
    files """
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

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
        # todo: this is just to stop code crashing on small runs
        if newborn_deaths == 0:
            newborn_deaths = 1
        if total_births_last_year == 0:
            total_births_last_year = 1

        # yearly number of complications
        sepsis = self.module.newborn_complication_tracker['early_onset_sepsis']
        mild_enceph = self.module.newborn_complication_tracker['mild_enceph']
        mod_enceph = self.module.newborn_complication_tracker['moderate_enceph']
        severe_enceph = self.module.newborn_complication_tracker['severe_enceph']
        all_enceph = mild_enceph + mod_enceph + severe_enceph
        ftt = self.module.newborn_complication_tracker['not_breathing_at_birth']
        death = self.module.newborn_complication_tracker['death']
        sepsis_death = self.module.newborn_complication_tracker['neonatal_sepsis_death']
        ftt_death = self.module.newborn_complication_tracker['not_breathing_at_birth_death']
        mild_enceph_death = self.module.newborn_complication_tracker['mild_enceph_death']
        moderate_enceph_death = self.module.newborn_complication_tracker['moderate_enceph_death']
        severe_enceph_death = self.module.newborn_complication_tracker['severe_enceph_death']
        preterm_birth_death = self.module.newborn_complication_tracker['preterm_birth_other_death']

        sepsis_treatment = self.module.newborn_complication_tracker['sep_treatment']
        resus = self.module.newborn_complication_tracker['resus']
    #    tetra_cycline = self.module.newborn_complication_tracker['t_e_d']

        dict_for_output = {'births': total_births_last_year,
                           'neonatal_deaths': newborn_deaths,
                           'checker_deaths': death,
                           'nmr_early': newborn_deaths/total_births_last_year * 1000,
                           'early_preterm_births': early_preterm_births,
                           'late_preterm_births': late_preterm_births,
                           'total_preterm_births': total_preterm_births,
                           'tptb_incidence': total_preterm_births/total_births_last_year * 100,
                           'preterm_birth_death': preterm_birth_death,
                           'low_birth_weight': low_birth_weight / total_births_last_year * 100,
                           'small_for_gestational_age': small_for_gestational_age / total_births_last_year * 100,
                           'sepsis_crude': sepsis,
                           'sepsis_incidence': sepsis / total_births_last_year * 100,
                           'sepsis_treatment_crude': sepsis_treatment,
                           'sepsis_deaths': sepsis_death,
                           'mild_enceph_incidence': mild_enceph / total_births_last_year * 100,
                           'mild_enceph_death': mild_enceph_death,
                           'mod_enceph_incidence': mod_enceph / total_births_last_year * 100,
                           'mod_enceph_death': moderate_enceph_death,
                           'severe_enceph_incidence': severe_enceph / total_births_last_year * 100,
                           'severe_enceph_death': severe_enceph_death,
                           'total_enceph_incidence': all_enceph / total_births_last_year * 100,
                           'total_enceph_death': mild_enceph_death + moderate_enceph_death + severe_enceph_death,
                           'ftt_crude': ftt,
                           'ftt_incidence': ftt / total_births_last_year * 100,
                           'ftt_death': ftt_death,
                           'resus_crude': resus,
                           # 'resus_rate': resus / ftt * 100,
                           }
        # TODO: health system outputs, check denominators

        logger.info(key='neonatal_summary_stats', data=dict_for_output, description='Yearly summary statistics output '
                                                                                    'from the neonatal outcome module')

        # Reset the EventTracker
        self.module.newborn_complication_tracker = {
            'early_init_bf': 0,
            'early_onset_sepsis': 0,
            'neonatal_sepsis_death': 0,
            'mild_enceph': 0,
            'mild_enceph_death': 0,
            'moderate_enceph': 0,
            'moderate_enceph_death': 0,
            'severe_enceph': 0,
            'severe_enceph_death': 0,
            'preterm_rds': 0,
            'respiratory_distress_death': 0,
            'congenital_anomaly_death': 0,
            'not_breathing_at_birth': 0,
            'not_breathing_at_birth_death': 0,
            'preterm_birth_other_death': 0,
            't_e_d': 0,
            'resus': 0,
            'sep_treatment': 0,
            'vit_k': 0,
            'death': 0}
