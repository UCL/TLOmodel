from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
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
        'base_incidence_sga': Parameter(
            Types.REAL, 'baseline incidence of small for gestational age for neonates'),
        'prob_congenital_ba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_cba_type': Parameter(
            Types.LIST, 'Probability of types of CBA'),
        'prob_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a neonate developing sepsis following birth'),
        'prob_resp_depression': Parameter(
            Types.REAL, 'baseline probability of a neonate developing intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'prob_encephalopathy': Parameter(
            Types.REAL, 'baseline probability of a neonate developing encephalopathy of any severity following birth'),
        'prob_enceph_severity': Parameter(
            Types.LIST, 'probability of the severity of encephalopathy in a newborn who is encephalopathic'),
        'prob_ivh_preterm': Parameter(
            Types.REAL, 'baseline probability of a preterm neonate developing an intravascular haemorrhage as a result'
                        'of prematurity '),
        'prob_nec_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing necrotising enterocolitis as a result of '
            'prematurity'),
        'prob_nrds_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing newborn respiratory distress syndrome as a result of'
            'prematurity '),
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
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_enceph_mild_mod': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to mild/moderate neonatal encephalopathy'),
        'cfr_enceph_severe': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to sever neonatal encephalopathy'),
        'cfr_respiratory_depression': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to respiratory depression'),
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
        'nb_respiratory_depression': Property(Types.BOOL, 'whether this neonate has been born asphyxiated and apneic '
                                                          'due to intrapartum related complications'),
        'nb_hypoxic_ischemic_enceph': Property(Types.BOOL, 'whether a perinatally asphyxiated neonate has developed '
                                                           'hypoxic ischemic encephalopathy'),
        'nb_encephalopathy': Property(Types.CATEGORICAL, 'None, mild encephalopathy, moderate encephalopathy, '
                                                         'severe encephalopathy',
                                      categories=['none', 'mild_enceph', 'moderate_enceph', 'severe_enceph']),
        'nb_intravascular_haem': Property(Types.BOOL, 'whether this neonate has developed an intravascular haemorrhage '
                                                      'following preterm birth'),
        'nb_necrotising_entero': Property(Types.BOOL, 'whether this neonate has developed necrotising enterocolitis '
                                                      'following preterm birth'),
        'nb_resp_distress_synd': Property(Types.BOOL, 'whether this neonate has developed newborn respiritory distress '
                                                      'syndrome following preterm birth '),
        'nb_retinopathy_prem': Property(Types.CATEGORICAL, 'Level of visual disturbance due to retinopathy of'
                                                           ' prematurity: None, mild, moderate, severe, blindness',
                                        categories=['none', 'mild', 'moderate', 'severe', 'blindness']),
        'nb_ongoing_impairment': Property(Types.CATEGORICAL, 'none, mild motor, mild motor and cognitive, '
                                                             'moderate motor, moderate motor and cognitive, '
                                                             'severe motor, severe motor and cognitive',
                                          categories=['none', 'mild_mot', 'mild_mot_cog', 'mod_mot', 'mod_mot_cog',
                                                      'severe_mot', ' severe_mot_cog']),
        'nb_birth_weight': Property(Types.CATEGORICAL, 'extremely low birth weight (<1000g), very low birth weight '
                                                       '(<1500g), low birth weight (<2500g),'
                                                       'normal birth weight (>2500g)',
                                    categories=['ext_LBW', 'very_LBW', 'LBW', 'NBW']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'small for gestational age, average for gestational'
                                                                   ' age, large for gestational age',
                                                categories=['SGA', 'AGA', 'LGA']),
        'nb_early_breastfeeding': Property(Types.BOOL, 'whether this neonate is exclusively breastfed after birth'),
        'nb_kmc': Property(Types.BOOL, 'whether this neonate received kangaroo mother care following birth'),
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

        params['nb_newborn_equations'] = {
            'lbw': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['base_incidence_low_birth_weight'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'sga': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['base_incidence_sga'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'sepsis': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_early_onset_neonatal_sepsis'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'resp_depression': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_resp_depression'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'encephalopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_encephalopathy'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'resp_distress_synd': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_nrds_preterm'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'intra_vent_haem': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_ivh_preterm'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'nec': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_nec_preterm'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'retinopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_retinopathy_preterm'],
             Predictor('age_years').when('.between(0,2)', 1))
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_congenital_anomaly'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_respiratory_depression'] = False
        df.loc[df.is_alive, 'nb_hypoxic_ischemic_enceph'] = False
        df.loc[df.is_alive, 'nb_encephalopathy'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_intravascular_haem'] = False
        df.loc[df.is_alive, 'nb_necrotising_entero'] = False
        df.loc[df.is_alive, 'nb_resp_distress_synd'] = False
        df.loc[df.is_alive, 'nb_retinopathy_prem'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_ongoing_impairment'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_birth_weight'].values[:] = 'NBW'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'].values[:] = 'AGA'
        df.loc[df.is_alive, 'nb_early_breastfeeding'] = False
        df.loc[df.is_alive, 'nb_kmc'] = False
        df.loc[df.is_alive, 'nb_death_after_birth'] = False
        df.loc[df.is_alive, 'nb_death_after_birth_date'] = pd.NaT

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        return self.rng.random_sample(size=1) < eq.predict(self.sim.population.props.loc[[person_id]])[person_id]

    def set_neonatal_death_status(self, individual_id, cause):
        """This function  is called for neonates that have experienced a complication after birth and determines if it
        will cause their death. Properties in the DF are set accordingly."""
        df = self.sim.population.props
        params = self.parameters

        if self.rng.random_sample() < params[f'cfr_{cause}']:
            df.at[individual_id, 'nb_death_after_birth'] = True
            df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

            logger.debug(F'This is NewbornOutcomes scheduling a death for person %d on date %s who died due to {cause}'
                         'complications following birth', individual_id, self.sim.date)

    def on_birth(self, mother_id, child_id):
        """The on_birth function of this module is used to apply the probability that a newborn will experience
        complications following delivery (which may or may not be attributable to the delivery process). This section of
        code is subject to change as it contains some dummy code (low birth weight/SGA application) and needs review by
        a clinician"""

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
        df.at[child_id, 'nb_respiratory_depression'] = False
        df.at[child_id, 'nb_hypoxic_ischemic_enceph'] = False
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_intravascular_haem'] = False
        df.at[child_id, 'nb_necrotising_entero'] = False
        df.at[child_id, 'nb_resp_distress_synd'] = False
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_ongoing_impairment'] = 'none'
        df.at[child_id, 'nb_birth_weight'] = 'NBW'
        df.at[child_id, 'nb_size_for_gestational_age'] = 'AGA'
        df.at[child_id, 'nb_early_breastfeeding'] = False
        df.at[child_id, 'nb_kmc'] = False
        df.at[child_id, 'nb_death_after_birth'] = False
        df.at[child_id, 'nb_death_after_birth_date'] = pd.NaT

        child = df.loc[child_id]

        # Here we set the variables for newborns delivered at less than 37 weeks, allocating them to either late or
        # early  preterm based on the gestation at labour

        if m['labour_state'] == 'EPTL':
            df.at[child_id, 'nb_early_preterm'] = True
            logger.info('%s|early_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        elif m['labour_state'] == 'LPTL':
            df.at[child_id, 'nb_late_preterm'] = True
            logger.info('%s|late_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})
        else:
            df.at[child_id, 'nb_early_preterm'] = False
            df.at[child_id, 'nb_late_preterm'] = False

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
            nci[child_id] = {'cord_care': False,
                             'bcg_vacc': False,
                             'polio_vacc': False,
                             'vit_k': False,
                             'tetra_eye_d': False,
                             'proph_abx': False,
                             'ongoing_sepsis_risk': mni[mother_id]['risk_newborn_sepsis']}

            # TODO:  review in context of properties

    # ============(dummy)================= BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE ===============(dummy)============

            # We apply an incidence of low birth weight and small for gestational age for preterm infants. This section
            # of code is place holding for code from the yet unfinished malnutrition module
            if ~child.nb_early_preterm & ~child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['lbw'], child_id):
                    df.at[child_id, 'nb_birth_weight'] = 'LBW'  # currently limiting to just LBW

            if child.nb_early_preterm:
                df.at[child_id, 'nb_birth_weight'] = 'LBW'

            # Magic number 0.7 is an arbitrary probability determining LBW in late preterm infants
            if child.nb_late_preterm & (self.rng.random_sample() < 0.7):
                df.at[child_id, 'nb_birth_weight'] = 'LBW'

            if self.eval(params['nb_newborn_equations']['sga'], child_id):
                df.at[child_id, 'nb_size_for_gestational_age'] = 'SGA'

    # ================================== COMPLICATIONS FOLLOWING BIRTH ================================================
            # Here, using linear equations, we determine individual newborn risk of complications following delivery.
            # This risk is either stored or applied depending on the complication, as detailed below

    # --------------------------------------------  EARLY ONSET SEPSIS  -----------------------------------------------

            # Underlying risk of newborn sepsis is stored in the maternal_and_newborn_info dictionary house in the
            # labour  module. Using a multiplicative model we apply the impact of riskfactors to this baseline rate,
            # which is currently incomplete. We then store their risk of sepsis to allow modification by treatment
            # effects (i.e. prophylactic antibiotics).
            if m['delivery_setting'] == 'facility_delivery':
                rf1 = 1
                riskfactors = rf1
                nci[child_id]['ongoing_sepsis_risk'] = riskfactors * nci[child_id]['ongoing_sepsis_risk']
                # TODO: this should be adapted to use the linear model if possible

            # If the neonate was born at home we calculate the risk and make changes to the data frame
            elif m['delivery_setting'] == 'home_birth' and self.eval(params['nb_newborn_equations']['sepsis'],
                                                                     child_id):
                df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True

                logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            child_id, self.sim.date)
                logger.info('%s|early_onset_nb_sep_hb|%s', self.sim.date, {'person_id': child_id})

    # -------------------------------------  RESPIRATORY DEPRESSION  ---------------------------------------------------

            # As with sepsis, we use the mni dictionary for the baseline risk. As preventative treatment effects for
            # respiratory depression will be handled during labour we modify the data frame here immediately
            if m['delivery_setting'] == 'facility_delivery':
                rf1 = 1
                riskfactors = rf1
                if self.rng.random_sample() < (riskfactors * mni[mother_id]['risk_newborn_ba']):
                    df.at[child_id, 'nb_respiratory_depression'] = True
                    logger.info('Neonate %d has been born asphyxiated in a health facility on date %s', child_id,
                                self.sim.date)
                    logger.info('%s|birth_asphyxia_fd|%s', self.sim.date, {'person_id': child_id})
                    # TODO: as with sepsis this should be modified to use the LM parameter.

                # If the neonate was born at home we calculate the risk and make changes to the data frame
                elif m['delivery_setting'] == 'home_birth':
                    if self.eval(params['nb_newborn_equations']['resp_depression'], child_id):
                        df.at[child_id, 'nb_respiratory_depression'] = True
                        logger.info('Neonate %d has been born asphyxiated following a home birth on date %s', child_id,
                                    self.sim.date)
                        logger.info('%s|birth_asphyxia_hb|%s', self.sim.date, {'person_id': child_id})

                # TODO: consider effect of cord prolapse- I think too granular

    # ------------------------------------  NEONATAL ENCEPHALOPATHY ---------------------------------------------------

            # We use the linear model to predict individual probability of neonatal encephalopathy and make the
            # changes to the data frame

            if self.rng.random_sample() < params['nb_newborn_equations']['encephalopathy']\
               .predict(df.loc[[child_id]])[child_id]:

                # For a newborn who is encephalopathic we then set the severity
                severity_enceph = self.rng.choice(('mild', 'moderate', 'severe'), p=params['prob_enceph_severity'])
                if severity_enceph == 'mild':
                    df.at[child_id, 'nb_encephalopathy'] = 'mild_enceph'
                elif severity_enceph == 'moderate':
                    df.at[child_id, 'nb_encephalopathy'] = 'moderate_enceph'
                else:
                    df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'

                #  TODO: To determine temporal direction of causal influence of neonatal encephalopathy on respiratory
                #   depression, risk of NE may need to be applied first

    # ================================== COMPLICATIONS IN NEONATES DELIVERED PRETERM  =================================

            # Using the linear model we determine individual risk of complications associated with prematurrity for
            # premature newborns. This code will be reviewed as we may not need this level of specificity. Similary we
            # have no code for treatment for these newborns yet

            if child.nb_early_preterm & (df.at[mother_id, 'ps_gestational_age_in_weeks'] < 32):
                if self.eval(params['nb_newborn_equations']['ivh'], child_id):
                    df.at[child_id, 'nb_intravascular_haem'] = True
                    logger.debug('Neonate %d has developed intravascular haemorrhage secondary to prematurity',
                                 child_id)

            if child.nb_early_preterm or child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['nec'], child_id):
                    df.at[child_id, 'nb_necrotising_entero'] = True
                    logger.debug('Neonate %d has developed necrotising enterocolitis secondary to prematurity',
                                 child_id)

                if self.eval(params['nb_newborn_equations']['resp_distress_synd'], child_id):
                    df.at[child_id, 'nb_resp_distress_synd'] = True
                    logger.debug(
                            'Neonate %d has developed newborn respiratory distress syndrome secondary to prematurity',
                            child_id)

                    if self.eval(params['nb_newborn_equations']['retinopathy'], child_id):
                        random_draw = self.rng.choice(('mild', 'moderate', 'severe', 'blindness'),
                                                      p=params['prob_retinopathy_severity'])
                        df.at[child_id, 'nb_retinopathy_prem'] = random_draw
                        logger.debug(f'Neonate %d has developed {random_draw }retinopathy of prematurity',
                                     child_id)

            # TODO: How will we apply the risk reduction of certain complications associated with steroids delivered
            #  antenatally

            # TODO: review lit r.e. retinopathy
            # TODO: Consider application of impairment variable and how based to decide probabilities

    # ===================================== DISABILITY/LONG TERM IMPAIRMENT  ==========================================

            # This is a placeholder for dealing with properties that will capture the long term outcome sequalae
            # associated with a number of these complications. This will likely be stored in the nb_ongoing_impairment
            # property

    # ======================================= SCHEDULING NEWBORN CARE  ================================================

                # Neonates who were delivered in a facility are automatically scheduled to receive care after birth
            if m['delivery_setting'] == 'facility_delivery':
                event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self, person_id=child_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))
                logger.debug(
                    'This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDelivery '
                    'for person %d following a facility delivery', child_id)

                # If this neonate was delivered at home and develops a complications we determine the likelihood of care
                # seeking
            if (m['delivery_setting'] == 'home_birth') & (child.nb_respiratory_depression or
                                                          child.nb_early_onset_neonatal_sepsis or
                                                          (child.nb_encephalopathy == 'mild_enceph') or
                                                          (child.nb_encephalopathy == 'moderate_enceph') or
                                                          (child.nb_encephalopathy == 'severe_enceph')):
                prob = 0.75
                random = self.rng.random_sample(size=1)
                if random < prob:
                    event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self.module, person_id=child_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=3)

                    logger.debug(
                            'This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDelivery'
                            'for person %d following a home birth', child_id)

                    # TODO: Above code needs to be reformatted, using ?symptom manager for careseeking in the instance
                    #  of emergencies at a home deliver

    # ============================================ BREAST FEEDING AT HOME ============================================
                # Using DHS data we apply a one of probability that women who deliver at home will initiate
                # breastfeeding within one hour of birth
            if (m['delivery_setting'] == 'home_birth') and (~child.nb_respiratory_depression and
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

            self.sim.schedule_event(NewbornDeathEvent(self, child_id), self.sim.date + DateOffset(days=2))
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

        # Using the set_neonatal_death_status function, defined above, it is determined if newborns who have experienced
        # complications will die because of them. Successful treatment in a HSI will turn off theses properties meaning
        # newborns will not die due to the complication. This logic needs to be reviewed

        if child.nb_early_onset_neonatal_sepsis:
            self.module.set_neonatal_death_status(individual_id, cause='neonatal_sepsis')

        if (child.nb_encephalopathy == 'mild_enceph') or (child.nb_encephalopathy == 'moderate_enceph'):
            self.module.set_neonatal_death_status(individual_id, cause='enceph_mild_mod')

        if child.nb_encephalopathy == 'severe_enceph':
            self.module.set_neonatal_death_status(individual_id, cause='enceph_severe')

        if child.nb_respiratory_depression:
            self.module.set_neonatal_death_status(individual_id, cause='respiratory_depression')

        child = df.loc[individual_id]
        if child.nb_death_after_birth:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)

        #  Todo: make sure this is delayed enough following HSI?
        #  TODO: Tim C suggested we need to create an offset (using a distribution?) so we're generating deaths for the
        #   first 48 hours
        # TODO: No deaths from preterm birth complications are handled here- need to determine where that will be done
        # TODO: will we treat unsuccessful BA and enceph as the same?
        # TODO: use append to add cause of death to mni?

# ================================ HEALTH SYSTEM INTERACTION EVENTS ================================================


class HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(HSI_Event, IndividualScopeEventMixin):
    """This HSI ReceivesCareFollowingDelivery. This event is scheduled by the on_birth function. All newborns whose
    mothers delivered in a facility are automatically scheduled to this event. Newborns delivered at home, but who
    experience complications, have this event scheduled due via a care seeking equation (currently just a dummy). It is
    responsible for prophylactic treatments following delivery (i.e. cord care, breastfeeding), applying risk of
    complications in facility and referral for additional treatment. This module will be reviewed with a clinician and
     may be changed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesCareFollowingDelivery'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        nci = self.module.newborn_care_info
        child = df.loc[person_id]

        logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery, neonate %d is receiving care from a '
                    'skilled birth attendant following their birth', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables
                                             ['Intervention_Pkg'] == 'Clean practices and immediate essential newborn '
                                                                     'care (in facility)',
                                                                     'Intervention_Pkg_Code'])[0]
        pkg_code_bcg = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'BCG vaccine', 'Intervention_Pkg_Code'])[0]
        pkg_code_polio = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Polio vaccine',
                                                                                      'Intervention_Pkg_Code'])[0]

        item_code_vk = pd.unique(
            consumables.loc[consumables['Items'] == 'vitamin K1  (phytomenadione) 1 mg/ml, 1 ml, inj._100_IDA',
                            'Item_Code'])[0]
        item_code_tc = pd.unique(
            consumables.loc[consumables['Items'] == 'tetracycline HCl 3% skin ointment, 15 g_10_IDA', 'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code: 1, pkg_code_bcg: 1, pkg_code_polio: 1},
            'Item_Code': {item_code_vk: 1, item_code_tc: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: Need to ensure not double counting consumables (i.e. chlorhexidine for cord care already included in
        #  delivery kit?)
        # TODO: chlorhex should be eye drops not ointment?
        # TODO: apply effect of squeeze factor

        # We apply the effect of a number of interventions that newborns in a facility will receive including cord care,
        # vaccinations, vitamin K prophylaxis, tetracycline eyedrops, kangaroo mother care and prophylactic antibiotics

# ----------------------------------- CHLORHEXIDINE CORD CARE ----------------------------------------------------------

        nci[person_id]['cord_care'] = True

# ------------------------------------- VACCINATIONS (BCG/POLIO) -------------------------------------------------------

        # For vaccines, vitamin K and tetracycline we condition these interventions  on the availibility of the
        # consumable
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_bcg]:
            logger.debug('pkg_code_bcg is available, so use it.')
            nci[person_id]['bcg_vacc'] = True
        else:
            logger.debug('pkg_code_bcg is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive a BCG vaccine as there was no stock available', person_id)
            nci[person_id]['bcg_vacc'] = False

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_polio]:
            logger.debug('pkg_code_polio is available, so use it.')
            nci[person_id]['polio_vacc'] = True
        else:
            logger.debug('pkg_code_polio is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive a BCG vaccine as there was no stock availble', person_id)
            nci[person_id]['polio_vacc'] = False

# ------------------------------------------ VITAMIN K  ----------------------------------------------------------------
        if outcome_of_request_for_consumables['Item_Code'][item_code_vk]:
            logger.debug('item_code_vk is available, so use it.')
            nci[person_id]['vit_k'] = True
        else:
            logger.debug('item_code_vk is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive vitamin K prophylaxis as there was no stock available', person_id)
            nci[person_id]['vit_k'] = False

# --------------------------------------- TETRACYCLINE EYE DROPS -------------------------------------------------------
        if outcome_of_request_for_consumables['Item_Code'][item_code_tc]:
            logger.debug('item_code_tc is available, so use it.')
            nci[person_id]['tetra_eye_d'] = True
        else:
            logger.debug('item_code_tc is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive tetracycline eyedrops as there was no stock availble', person_id)
            nci[person_id]['tetra_eye_d'] = False

# --------------------------------- ANTIBIOTIC PROPHYLAXIS (MATERNAL RISK FACTORS)-------------------------------------

        # TODO: Confirm the guidelines (indications) and consumables for antibiotic prophylaxis
        # nci[person_id]['proph_abx'] = True

# ----------------------------------------- KANGAROO MOTHER CARE -------------------------------------------------------

        # Here we use a probability, derived from the DHS, to determine if a woman with a low birth weight infant will
        # be encouraged to undertake KMC. Currently only 'stable' (no complication) newborns can undergo KMC. This wil
        # need review
        if (child.nb_birth_weight == 'LBW') and (~child.nb_respiratory_depression and
                                                 ~child.nb_early_onset_neonatal_sepsis
                                                 and (child.nb_encephalopathy == 'none')):

            if self.module.rng.random_sample() < params['prob_facility_offers_kmc']:
                df.at[person_id, 'nb_kmc'] = True
            # TODO: evidence suggests KMC reduces the risk of sepsis and this needs to be applied
            # TODO: Check guidelines regarding if KMC is only used in stable infants

# ------------------------------ EARLY INITIATION OF BREAST FEEDING ----------------------------------------------------

        # As with KMC we use a DHS derived probability that a woman who delivers in a facility will initiate
        # breastfeeding within one hour
        if self.module.rng.random_sample() < params['prob_early_breastfeeding_hf']:
            df.at[person_id, 'nb_early_breastfeeding'] = True
            logger.debug('Neonate %d has started breastfeeding within 1 hour of birth', person_id)
        else:
            logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', person_id)

# ------------------------------ RECALCULATE SEPSIS RISK---------------------------------------------------------------

        # Following the application of the prophylactic/therapeutic effect of these interventions we then recalculate
        # individual sepsis risk and determine if this newborn will develop sepsis

        if self.module.rng.random_sample() < nci[person_id]['ongoing_sepsis_risk']:
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True

            logger.debug('Neonate %d has developed early onset sepsis in a health facility on date %s', person_id,
                         self.sim.date)
            logger.debug('%s|early_onset_nb_sep_fd|%s', self.sim.date, {'person_id': person_id})
            # TODO code in effect of prophylaxis

#  ================================ SCHEDULE ADDITIONAL TREATMENT ===================================================

        # Finally, for newborns who have experienced a complication within a facility, additional treatment is scheduled
        # through other HSIs
        if child.nb_respiratory_depression:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling resuscitation for '
                        'neonate %d who has experienced birth asphyxia following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if child.nb_early_onset_neonatal_sepsis:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling treatment for neonate'
                        'person %d who has developed early onset sepsis following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass


class HSI_NewbornOutcomes_ReceivesNewbornResuscitation(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI ReceivesNewbornResuscitation. This event is scheduled by HSI ReceivesCareFollowingDelivery if a
    child experiences respiratory depression. This event contains the intervention basic newborn resuscitation. This
    event is unfinished and will need to schedule very sick neonates for additional inpatient care"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesNewbornResuscitation'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_NewbornOutcomes_ReceivesNewbornResuscitation, neonate %d is receiving newborn '
                    'resuscitation following birth ', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Neonatal resuscitation '
                                                                                '(institutional)',
                                             'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: apply effect of squeeze factor

        # The block of code below is the intervention, newborns can only be resuscitated if the consumables are
        # available, so the effect is conditioned on this
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('resuscitation equipment is available, so use it.')
            if self.module.rng.random_sample() < params['prob_successful_resuscitation']:
                df.at[person_id, 'nb_respiratory_depression'] = False
                logger.info('Neonate %d has been successfully resuscitated after delivery with birth asphyxia',
                            person_id)

        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: apply a probability of death without resuscitation here?
            # TODO: schedule additional care for very sick newborns

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass


class HSI_NewbornOutcomes_ReceivesTreatmentForSepsis(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI ReceivesTreatmentForSepsis. This event is scheduled by HSI ReceivesCareFollowingDelivery if a
       child experiences early on sent neonatal sepsis. This event contains the intervention intravenous antibiotics.
       This  event is unfinished and will need to schedule very sick neonates for additional inpatient care"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesTreatmentForSepsis'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_NewbornOutcomes_ReceivesTreatmentForSepsis, neonate %d is receiving treatment '
                    'for early on set neonatal sepsis following birth ', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sep = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] == 'Treatment of local infections (newborn)',
                                                 'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # TODO: Condition intervention on availability of antibiotics, considering 1st/2nd/3rd line and varying
        #  efficacy?

        # Here we use the treatment effect to determine if the newborn remains septic. This logic will need to be
        # changed to reflect need for inpatient admission and longer course of antibiotic
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            if params['prob_cure_antibiotics'] > self.module.rng.random_sample():
                df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = False
        else:
            logger.debug('pkg_code_sepsis is not available, so can' 't use it.')

        # TODO: septic newborns will receive a course of ABX as an inpatient, this needs to be scheduled here

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass


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
